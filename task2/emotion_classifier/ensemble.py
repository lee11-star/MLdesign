"""Probability fusion with explicit sample identity checks."""
import numpy as np


def align_probabilities(reference_ids, ids, probabilities, reference_targets=None, targets=None):
    reference_ids, ids = list(reference_ids), list(ids)
    if len(set(ids)) != len(ids) or len(set(reference_ids)) != len(reference_ids) or set(ids) != set(reference_ids):
        raise ValueError('Ensemble sample IDs must match exactly without duplicates')
    if np.asarray(probabilities).shape != (len(ids), 6):
        raise ValueError('Expected six probabilities per sample')
    positions = {image_id: i for i, image_id in enumerate(ids)}
    order = [positions[image_id] for image_id in reference_ids]
    if reference_targets is not None and (targets is None or not np.array_equal(reference_targets, np.asarray(targets)[order])):
        raise ValueError('Ensemble validation labels differ after ID alignment')
    return np.asarray(probabilities)[order]


def fuse_probabilities(probabilities, weights, temperatures=None):
    weights = np.asarray(weights, dtype=np.float64)
    temperatures = np.ones(len(weights)) if temperatures is None else np.asarray(temperatures, dtype=np.float64)
    if (len(probabilities) != len(weights) or len(temperatures) != len(weights) or
            not np.isfinite(weights).all() or not np.isfinite(temperatures).all() or
            (weights < 0).any() or weights.sum() <= 0 or (temperatures <= 0).any()):
        raise ValueError('Invalid fusion weights or temperatures')
    result = np.zeros_like(probabilities[0], dtype=np.float64)
    for probs, weight, temperature in zip(probabilities, weights / weights.sum(), temperatures):
        probs = np.asarray(probs, dtype=np.float64)
        if probs.shape != result.shape or not np.isfinite(probs).all() or (probs < 0).any():
            raise ValueError('Invalid probability array')
        if not np.allclose(probs.sum(1), 1, atol=1e-5):
            raise ValueError('Probability rows must sum to one')
        # This softens saved (possibly TTA-averaged) probabilities; it is not fitted calibration.
        softened = np.clip(probs, 1e-12, 1) ** (1.0 / temperature)
        softened /= softened.sum(1, keepdims=True)
        result += weight * softened
    return result


def fit_temperatures(probabilities, targets):
    """Fit one scalar to each saved probability array, including averaged TTA."""
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    rows = np.arange(len(targets))
    temperatures = []
    for probs in probabilities:
        logs = np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1))

        def loss(log_temperature):
            scaled = logs / np.exp(log_temperature)
            return float((logsumexp(scaled, axis=1) - scaled[rows, targets]).mean())

        result = minimize_scalar(loss, bounds=(np.log(0.5), np.log(4.0)), method='bounded',
                                 options={'xatol': 1e-5})
        if not result.success:
            raise RuntimeError('Probability temperature fitting did not converge')
        temperatures.append(float(np.exp(result.x)))
    return np.asarray(temperatures)


def fit_simplex_weights(probabilities, targets, initial_weights):
    """Minimize ordinary cross entropy with nonnegative weights summing to one."""
    from scipy.optimize import minimize

    initial = np.asarray(initial_weights, dtype=np.float64)
    initial = initial / initial.sum()
    true_class = np.stack([np.asarray(p)[np.arange(len(targets)), targets] for p in probabilities], axis=1)

    def objective(weights):
        mixture = np.clip(true_class @ weights, 1e-12, 1)
        return float(-np.log(mixture).mean()), -(true_class / mixture[:, None]).mean(axis=0)

    result = minimize(objective, initial, jac=True, method='SLSQP',
                      bounds=[(0.0, 1.0)] * len(initial),
                      constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1,
                                   'jac': lambda w: np.ones_like(w)},
                      options={'ftol': 1e-10, 'maxiter': 200})
    if not result.success:
        raise RuntimeError(f'Fusion weight fitting did not converge: {result.message}')
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()


def fit_accuracy_weights(probabilities, targets, initial_weights):
    """Small coordinate search: correct-count first, cross entropy breaks ties."""
    arrays = np.stack([fuse_probabilities([p], [1.0]) for p in probabilities])
    weights = np.asarray(initial_weights, dtype=np.float64).copy()
    weights /= weights.sum()
    mixture = np.einsum('m,mnc->nc', weights, arrays)
    rows = np.arange(len(targets))

    def score(probs):
        return (int((probs.argmax(1) == targets).sum()),
                float(np.log(np.clip(probs[rows, targets], 1e-12, 1)).mean()))

    current = score(mixture)
    for step in (0.05, 0.025, 0.01):
        for _ in range(6):
            best, move = current, None
            for donor in range(len(weights)):
                amount = min(step, weights[donor])
                if amount <= 1e-12:
                    continue
                for receiver in range(len(weights)):
                    if receiver == donor:
                        continue
                    candidate = mixture + amount * (arrays[receiver] - arrays[donor])
                    candidate_score = score(candidate)
                    if candidate_score > best:
                        best, move = candidate_score, (donor, receiver, amount, candidate)
            if move is None:
                break
            donor, receiver, amount, mixture = move
            weights[donor] -= amount
            weights[receiver] += amount
            current = best
    return weights / weights.sum()

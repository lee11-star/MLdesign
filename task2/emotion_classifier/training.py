import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import CLASS_TO_LABEL, save_json
from .models import build_model, preprocess, preprocessing_config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    generator.set_state(state['loader'].cpu())


def make_loader(x, y, config, shuffle=False, generator=None):
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=config['batch_size'], shuffle=shuffle,
                      num_workers=config.get('num_workers', 0),
                      pin_memory=str(config['device']).startswith('cuda'), generator=generator)


@torch.inference_mode()
def evaluate(model, loader, architecture, device, tta=False, image_size=None):
    model.eval()
    probabilities, targets = [], []
    loss_sum = torch.zeros((), device=device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = model(preprocess(images, architecture, image_size=image_size))
            loss_sum += F.cross_entropy(logits.float(), labels, reduction='sum')
            probs = logits.float().softmax(dim=1)
            if tta:
                other = model(preprocess(images, architecture, flip=True, image_size=image_size)).float().softmax(dim=1)
                probs = (probs + other) / 2
        probabilities.append(probs.cpu().numpy())
        targets.append(labels.cpu().numpy())
    probs, labels = np.concatenate(probabilities), np.concatenate(targets)
    return dict(loss=float(loss_sum.cpu()) / len(labels), accuracy=float((probs.argmax(1) == labels).mean())), probs


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def save_run_metadata(checkpoint, output, has_validation):
    history = checkpoint['history']
    if history:
        with (output / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[-1]))
            writer.writeheader()
            writer.writerows(history)
    save_json(checkpoint['timings'], output / 'timings.json')
    save_json(dict(completed_epochs=checkpoint['epoch'], planned_epochs=checkpoint['config']['epochs'],
                   best_epoch=checkpoint['best_epoch'],
                   best_val_accuracy=checkpoint['best_accuracy'] if has_validation else None,
                   final_training=not has_validation,
                   last_epoch_seconds=checkpoint['timings'][-1]['seconds'] if checkpoint['timings'] else None),
              output / 'summary.json')


def recover_selected_artifact(checkpoint, output, has_validation):
    expected_epoch = checkpoint['best_epoch'] if has_validation else checkpoint['epoch']
    path = output / ('best.pt' if has_validation else 'final.pt')
    selected = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else None
    if selected is not None and selected['epoch'] == expected_epoch and selected['config'] == checkpoint['config']:
        return
    if expected_epoch != checkpoint['epoch']:
        raise ValueError('Missing/inconsistent earlier best checkpoint; latest checkpoint cannot reconstruct its weights')
    atomic_save(checkpoint, path)


def train_run(config, train_x, train_y, val_x, val_y, output, resume=None, stop_after=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists() and not resume:
        raise ValueError('Run already has a checkpoint; use --resume or a new output directory')
    config = dict(config)
    mixup_alpha = config.get('mixup_alpha', 0.0)
    mixup_probability = config.get('mixup_probability', 0.5)
    if mixup_alpha < 0 or not 0 <= mixup_probability <= 1:
        raise ValueError('MixUp requires alpha >= 0 and probability in [0, 1]')
    config['preprocessing'] = preprocessing_config(config['model'], config.get('image_size'),
                                                  config.get('geometry', 'standard'))
    config['class_to_label'] = CLASS_TO_LABEL
    seed_everything(config['seed'])
    device = torch.device(config['device'])
    model = build_model(config['model'], config.get('pretrained') if not resume else None,
                        head_dropout=config.get('head_dropout', 0.0),
                        trainable_blocks=config.get('trainable_blocks', 4),
                        dinov2_repo=config.get('dinov2_repo')).to(device)
    if config.get('initial_head') and not resume:
        model.classifier.load_state_dict(torch.load(config['initial_head'], map_location='cpu', weights_only=True), strict=True)
    backbone_scale = config.get('backbone_lr_scale', 1.0)
    if backbone_scale <= 0:
        raise ValueError('Backbone learning-rate scale must be positive')
    if backbone_scale == 1.0:
        parameters = model.parameters()
    else:
        head = model.fc if config['model'] == 'resnet18' else model.classifier
        head_parameters = list(head.parameters())
        if config['model'] == 'dan_confidence':
            head_parameters += list(model.confidence.parameters())
        head_ids = {id(p) for p in head_parameters}
        parameters = [dict(params=head_parameters, lr=config['lr'], name='head'),
                      dict(params=[p for p in model.parameters() if id(p) not in head_ids and p.requires_grad],
                           lr=config['lr'] * backbone_scale, name='backbone')]
    optimizer = torch.optim.AdamW(parameters, lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get('scheduler_epochs', config['epochs']))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    generator = torch.Generator().manual_seed(config['seed'])
    train_loader = make_loader(train_x, train_y, config, True, generator)
    val_loader = make_loader(val_x, val_y, config) if val_x is not None else None
    start, best_accuracy, best_epoch, history, timings = 0, -1.0, 0, [], []
    if resume:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Resume configuration differs from checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start, best_accuracy, best_epoch = checkpoint['epoch'], checkpoint['best_accuracy'], checkpoint['best_epoch']
        history, timings = checkpoint['history'], checkpoint['timings']
        restore_rng(checkpoint['rng'], generator)
        recover_selected_artifact(checkpoint, output, val_loader is not None)
        save_run_metadata(checkpoint, output, val_loader is not None)
    save_json(config, output / 'config.json')
    print(f"Training {config['model']} on {device}: train={len(train_y)}, val={0 if val_y is None else len(val_y)}, start={start}, epochs={config['epochs']}", flush=True)
    stop = min(config['epochs'], stop_after) if stop_after is not None else config['epochs']
    for epoch in range(start, stop):
        tick = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        confidence_training = config['model'] == 'dan_confidence'
        if confidence_training:
            classification_sum = torch.zeros((), device=device)
            ranking_sum = torch.zeros((), device=device)
        current_lr = optimizer.param_groups[0]['lr']
        current_backbone_lr = optimizer.param_groups[-1]['lr']
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                inputs = preprocess(images, config['model'], training=True,
                                    image_size=config['preprocessing']['size'],
                                    geometry=config.get('geometry', 'standard'))
                mixing = mixup_alpha > 0 and torch.rand((), device=device).item() < mixup_probability
                if mixing:
                    concentration = inputs.new_tensor(mixup_alpha)
                    proportion = torch.distributions.Beta(concentration, concentration).sample()
                    order = torch.randperm(len(labels), device=device)
                    inputs = proportion * inputs + (1 - proportion) * inputs[order]
                if confidence_training:
                    logits, attention = model.forward_with_confidence(inputs)
                else:
                    logits = model(inputs)
                smoothing = config.get('label_smoothing', 0.0)
                loss = F.cross_entropy(logits, labels, label_smoothing=smoothing)
                if mixing:
                    loss = proportion * loss + (1 - proportion) * F.cross_entropy(
                        logits, labels[order], label_smoothing=smoothing)
                if confidence_training:
                    from .facial import confidence_rank_loss
                    classification_loss = loss
                    rank_loss = confidence_rank_loss(attention, config.get('confidence_fraction', 0.7),
                                                     config.get('confidence_margin', 0.15))
                    loss = loss + config.get('confidence_regularization', 1.0) * rank_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().float() * len(labels)
            if confidence_training:
                classification_sum += classification_loss.detach().float() * len(labels)
                ranking_sum += rank_loss.detach().float() * len(labels)
            prediction = logits.detach().argmax(1)
            if mixing:
                correct += proportion * (prediction == labels).sum() + (1 - proportion) * (prediction == labels[order]).sum()
            else:
                correct += (prediction == labels).sum()
            if step and step % 60 == 0:
                print(f'  epoch {epoch + 1} batch {step}/{len(train_loader)}', flush=True)
        row = dict(epoch=epoch + 1, lr=current_lr, train_loss=float(loss_sum.cpu()) / len(train_y),
                   train_accuracy=float(correct.cpu()) / len(train_y))
        if confidence_training:
            row.update(train_classification_loss=float(classification_sum.cpu()) / len(train_y),
                       train_confidence_rank_loss=float(ranking_sum.cpu()) / len(train_y))
        if backbone_scale != 1.0:
            row['backbone_lr'] = current_backbone_lr
        if val_loader:
            metrics, _ = evaluate(model, val_loader, config['model'], device,
                                  image_size=config['preprocessing']['size'])
            row.update(val_loss=metrics['loss'], val_accuracy=metrics['accuracy'])
            if row['val_accuracy'] > best_accuracy:
                best_accuracy, best_epoch = row['val_accuracy'], epoch + 1
        else:
            best_epoch = epoch + 1
        scheduler.step()
        history.append(row)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        timings.append(dict(epoch=epoch + 1, seconds=seconds))
        checkpoint = dict(model=model.state_dict(), config=config, class_to_label=CLASS_TO_LABEL,
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                          epoch=epoch + 1, best_accuracy=best_accuracy, best_epoch=best_epoch,
                          history=history, timings=timings, rng=rng_state(generator))
        # last.pt is authoritative. If selected publication is interrupted at a
        # new best/final epoch, resume reconstructs that artifact from last.pt.
        atomic_save(checkpoint, output / 'last.pt')
        if best_epoch == epoch + 1:
            atomic_save(checkpoint, output / ('best.pt' if val_loader else 'final.pt'))
        save_run_metadata(checkpoint, output, val_loader is not None)
        print(f"epoch {epoch + 1}/{config['epochs']} loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f}" +
              (f" val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} best={best_accuracy:.4f}" if val_loader else '') +
              f' seconds={seconds:.1f}', flush=True)
    return history

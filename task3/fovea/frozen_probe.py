"""Frozen, label-independent features and a pixel-coordinate ridge head."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def design_matrix(features, sizes):
    features = np.asarray(features, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    camera = np.broadcast_to(sizes[:, None, :]/1000, (*features.shape[:2], 2))
    return np.concatenate([features, camera], axis=-1).reshape(-1, features.shape[-1]+2)


def fit_probe(features, frame, alpha):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from .supported_decoder import training_region

    sizes = frame[['width', 'height']].to_numpy()
    original = frame[['Fovea_X', 'Fovea_Y']].to_numpy()
    mirrored = original.copy()
    mirrored[:, 0] = sizes[:, 0]-mirrored[:, 0]
    target = np.stack([original, mirrored], axis=1).reshape(-1, 2)
    x = design_matrix(features, sizes)
    scaler = StandardScaler()
    with threadpool_limits(limits=4):
        standardized = scaler.fit_transform(x)
        ridge = Ridge(alpha=alpha).fit(standardized, target)
    coef = ridge.coef_/scaler.scale_[None]
    return {'coef': coef, 'intercept': ridge.intercept_-coef@scaler.mean_,
            'region': np.asarray(training_region(frame)), 'alpha': np.asarray(alpha)}


def predict_probe(head, features, sizes):
    sizes = np.asarray(sizes, dtype=np.float64)
    predicted = (design_matrix(features, sizes)@head['coef'].T+head['intercept']).reshape(len(sizes), 2, 2)
    predicted[:, 1, 0] = sizes[:, 0]-predicted[:, 1, 0]
    xy = predicted.mean(1)
    region = head['region']
    xy = np.clip(xy, sizes*region[:2], sizes*region[2:])
    return np.clip(xy, 0, sizes-1)


def cached_features(path, ids):
    with np.load(path) as cached:
        lookup = {int(value): i for i, value in enumerate(cached['ids'])}
        return cached['features'][[lookup[int(value)] for value in ids]]


def extract_features(frame, output, project_root):
    import torch
    from torch.nn import functional as F
    from .training import image_tensor, seed_everything

    output, root = Path(output), Path(project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with np.load(output) as cached:
            assert cached['ids'].tolist() == frame.id.tolist()
            assert cached['features'].shape[:2] == (len(frame), 2)
            assert np.isfinite(cached['features']).all()
        return output
    seed_everything(42)
    cache = root.parent/'task1/outputs/torch_cache/hub'
    source = cache/'facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8'
    weights = cache/'checkpoints/dinov2_vits14_pretrain.pth'
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    try:
        from dinov2.hub.backbones import dinov2_vits14
        model = dinov2_vits14(pretrained=False)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        model.requires_grad_(False).eval().cuda()
        images = image_tensor(frame.path, 392, torch.device('cuda'))
        mean = torch.tensor([.485, .456, .406], device='cuda').view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225], device='cuda').view(1, 3, 1, 1)
        features = []
        with torch.inference_mode():
            for start in range(0, len(frame), 8):
                views = []
                for flip in [False, True]:
                    batch = images[start:start+8].float()/255
                    if flip:
                        batch = batch.flip(-1)
                    with torch.autocast('cuda'):
                        layers = model.get_intermediate_layers((batch-mean)/std, n=4, return_class_token=True)
                        patch = layers[-1][0]
                        grid = patch.transpose(1, 2).reshape(len(batch), model.embed_dim, 28, 28)
                        feature = torch.cat([item[1] for item in layers]+[patch.mean(1), F.adaptive_avg_pool2d(grid, 4).flatten(1)], dim=1)
                    views.append(feature.float().cpu().numpy())
                features.append(np.stack(views, axis=1))
                print(f'Frozen DINO features {min(start+8, len(frame))}/{len(frame)}', flush=True)
        values = np.concatenate(features)
        np.savez_compressed(output, ids=frame.id.to_numpy(), features=values)
        metadata = {'encoder': 'original DINOv2 ViT-S/14 pretrained, frozen', 'input_size': 392,
                    'views': ['original', 'horizontal_flip'], 'feature_dim': int(values.shape[-1]),
                    'ids': frame.id.tolist(), 'labels_used_in_feature_extraction': False,
                    'source': str(source), 'weights': str(weights),
                    'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest()}
        output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    finally:
        sys.path.remove(str(source))
        sys.dont_write_bytecode = previous
    return output

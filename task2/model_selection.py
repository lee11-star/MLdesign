"""Run three controlled ablations, select a small fusion, retrain and validate a CSV."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from emotion_classifier.data import (CLASS_TO_LABEL, CLASSES, DEFAULT_DATA, DEFAULT_TEMPLATE,
                                     check_split, load_arrays, read_manifest, save_json)
from emotion_classifier.ensemble import align_probabilities, fuse_probabilities
from emotion_classifier.inference import load_checkpoint, predict_probabilities, write_submission
from evaluate import run_evaluation
from baseline_training import command, completed_run_is_consistent

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/optimization'


def check_requested_config(saved, requested):
    for key in ('model', 'epochs', 'scheduler_epochs', 'lr', 'weight_decay', 'seed', 'batch_size',
                'final_training', 'data_signature', 'label_smoothing', 'image_size', 'mixup_alpha', 'mixup_probability',
                'head_dropout', 'backbone_lr_scale', 'trainable_blocks', 'dinov2_repo',
                'initial_head', 'initial_head_signature', 'geometry',
                'confidence_fraction', 'confidence_margin', 'confidence_regularization'):
        default = {'label_smoothing': 0.0, 'mixup_alpha': 0.0, 'mixup_probability': 0.5,
                   'head_dropout': 0.0, 'backbone_lr_scale': 1.0, 'trainable_blocks': 4,
                   'geometry': 'standard', 'confidence_fraction': 0.7,
                   'confidence_margin': 0.15, 'confidence_regularization': 1.0}.get(key)
        old, new = saved.get(key, default), requested.get(key, default)
        if key == 'image_size':
            old = old or (48 if saved['model'] == 'cnn' else 224 if saved['model'] in ('dinov2_vits14', 'dan', 'dan_confidence') else 96)
            new = new or (48 if requested['model'] == 'cnn' else 224 if requested['model'] in ('dinov2_vits14', 'dan', 'dan_confidence') else 96)
        if old != new:
            raise ValueError(f'Existing experiment differs in {key}: {old!r} vs {new!r}; use a new directory')


def train_stage(output, requested, device):
    output.mkdir(parents=True, exist_ok=True)
    last = output / 'last.pt'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        check_requested_config(state['config'], requested)
        if completed_run_is_consistent(output, requested['epochs'], requested['final_training']):
            print(f'{output.name}: verified complete; reusing', flush=True)
            return
    args = ['train.py', '--model', requested['model'], '--epochs', str(requested['epochs']),
            '--scheduler-epochs', str(requested['scheduler_epochs']), '--lr', str(requested['lr']),
            '--weight-decay', str(requested['weight_decay']), '--label-smoothing', str(requested.get('label_smoothing', 0)),
            '--image-size', str(requested.get('image_size') or (48 if requested['model'] == 'cnn' else 224 if requested['model'] in ('dinov2_vits14', 'dan', 'dan_confidence') else 96)),
            '--batch-size', str(requested['batch_size']), '--seed', str(requested['seed']),
            '--mixup-alpha', str(requested.get('mixup_alpha', 0.0)),
            '--mixup-probability', str(requested.get('mixup_probability', 0.5)),
            '--head-dropout', str(requested.get('head_dropout', 0.0)),
            '--geometry', requested.get('geometry', 'standard'),
            '--backbone-lr-scale', str(requested.get('backbone_lr_scale', 1.0)),
            '--device', device, '--output', str(output)]
    if requested['model'] == 'dinov2_vits14':
        args += ['--trainable-blocks', str(requested.get('trainable_blocks', 4))]
        if requested.get('dinov2_repo'):
            args += ['--dinov2-repo', requested['dinov2_repo']]
    if requested.get('initial_head'):
        args += ['--initial-head', requested['initial_head']]
    if requested['model'] == 'dan_confidence':
        args += ['--confidence-fraction', str(requested.get('confidence_fraction', 0.7)),
                 '--confidence-margin', str(requested.get('confidence_margin', 0.15)),
                 '--confidence-regularization', str(requested.get('confidence_regularization', 1.0))]
    if requested.get('pretrained'):
        args += ['--pretrained', requested['pretrained']]
    if requested['final_training']:
        args.append('--full-data')
    if last.exists():
        args += ['--resume', str(last)]
    command(args, output / 'console.log')
    if not completed_run_is_consistent(output, requested['epochs'], requested['final_training']):
        raise ValueError(f'Incomplete training artifacts: {output}')
    state = torch.load(last, map_location='cpu', weights_only=False)
    check_requested_config(state['config'], requested)


def probability_metrics(probs, labels):
    pred = probs.argmax(1)
    return dict(accuracy=float((pred == labels).mean()),
                cross_entropy=float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
                per_class={name: float((pred[labels == i] == i).mean()) for i, name in enumerate(CLASSES)})


def write_report(models, comparison, selection, submission=None):
    lines = ['# 第二轮优化实验', '', '使用原有 5,633 张分层分组验证图像、同一划分和 seed=42。',
             '三组新增实验各训练 30 轮，输入统一为 96×96；分别只改变标签平滑、权重衰减或架构。',
             '训练使用标签平滑时，训练损失与普通交叉熵不同；验证损失始终为未平滑交叉熵。', '',
             '| 配置 | 最佳轮次 | 普通验证 | 翻转 TTA | 普通验证交叉熵 |',
             '|---|---:|---:|---:|---:|']
    for name, model in models.items():
        m = model['metrics']
        lines.append(f"| {name} | {m['best_epoch']} | {m['accuracy']:.2%} | {m['flip_tta_accuracy']:.2%} | {m['cross_entropy']:.4f} |")
    lines += ['', '融合候选限于单模型、既有 CNN+ResNet18 等权方案、最强单模型与 CNN/次强模型的等权组合，',
              '以及前三个单模型的等权组合。双模型额外比较将第一项概率按 T=2 软化。',
              '该变换作用于已平均的概率，并非拟合出的置信度校准。全部候选见 comparison.json。', '',
              f"选中验证方案：{selection['name']}，准确率 **{selection['accuracy']:.2%}**，交叉熵 {selection['cross_entropy']:.4f}。", '',
              '| 类别 | 原 ResNet18 TTA | 选中方案 |', '|---|---:|---:|']
    baseline = next(row for row in comparison if row['name'] == 'single:resnet18_transfer')
    for name in CLASSES:
        lines.append(f"| {name} | {baseline['per_class'][name]:.2%} | {selection['per_class'][name]:.2%} |")
    lines += ['', '上述准确率使用同一验证集选择最佳轮次、TTA 和融合方案，属于探索结果；没有新的独立验证集或 Kaggle 分数。',
              '本轮未进行 128×128、MixUp、多种子或新增类别权重实验。']
    if submission:
        lines += ['', '选中方案各成员使用清洗后的全部 28,165 张训练图像重训，轮数取其验证最佳轮次，',
                  '余弦调度总周期沿用验证阶段；融合权重、温度和 TTA 固定。',
                  f"生成 submission_optimized.csv，共 {submission['rows']:,} 行，模板 ID 顺序及整数 0–5 校验通过。", 
                  '全量模型没有独立验证准确率。原 submission.csv 保留，未上传 Kaggle。']
    (ROOT / 'optimization_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda')
    p.add_argument('--validation-only', action='store_true')
    args = p.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from task2')
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'outputs/data/manifest.csv'
    signature = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = read_manifest(manifest)
    check_split(rows)
    _, labels, reference_ids = load_arrays(rows, DEFAULT_DATA, 'val', 'outputs/data')
    reference_ids = np.asarray(reference_ids)
    baseline = json.loads((ROOT / 'outputs/resnet18_transfer/config.json').read_text(encoding='utf-8'))
    candidates = [('resnet18_smoothing', dict(baseline, label_smoothing=0.1)),
                  ('resnet18_weight_decay', dict(baseline, weight_decay=1e-3)),
                  ('convnext_tiny', dict(baseline, model='convnext_tiny', pretrained=None))]
    models = {}
    stages = [('cnn_baseline', ROOT / 'outputs/cnn_baseline', None),
              ('resnet18_transfer', ROOT / 'outputs/resnet18_transfer', None)]
    stages += [(name, OUT / name, config) for name, config in candidates]
    for name, output, requested in stages:
        save_json(dict(stage='validation', current_run=name), OUT / 'status.json')
        if requested:
            train_stage(output, requested, args.device)
        config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
        if config['data_signature'] != signature or config['class_to_label'] != CLASS_TO_LABEL:
            raise ValueError('Validation run uses a different split or class mapping')
        metric = run_evaluation(output / 'best.pt', device=args.device)
        with np.load(output / 'evaluation/validation_probabilities.npz') as data:
            key = 'flip_tta_probabilities' if metric['selected_tta'] else 'probabilities'
            probs = align_probabilities(reference_ids, data['ids'], data[key], labels, data['targets'])
        models[name] = dict(config=config, metrics=metric, probabilities=probs, output=output)
        gc.collect()
        torch.cuda.empty_cache()
    comparison = []
    fused = {}

    def add(name, members, temperatures):
        weights = [1 / len(members)] * len(members)
        probs = fuse_probabilities([models[item]['probabilities'] for item in members], weights, temperatures)
        entry = dict(name=name, members=members, weights=weights, temperatures=temperatures,
                     tta=[models[item]['metrics']['selected_tta'] for item in members],
                     **probability_metrics(probs, labels))
        comparison.append(entry)
        fused[name] = probs

    for name in models:
        add('single:' + name, [name], [1.0])
    ranked = sorted(models, key=lambda name: models[name]['metrics']['selected_accuracy'], reverse=True)
    pairs = [('resnet18_transfer', 'cnn_baseline'), (ranked[0], 'cnn_baseline'), (ranked[0], ranked[1])]
    for first, second in dict.fromkeys(pairs):
        if first == second:
            continue
        for temperature in [1.0, 2.0]:
            add(f'{first}+{second}:T={temperature:g}', [first, second], [temperature, 1.0])
    add('top3_equal', ranked[:3], [1.0] * 3)
    selected = max(comparison, key=lambda entry: (entry['accuracy'], -entry['cross_entropy']))
    selected = dict(selected, data_signature=signature, samples=len(labels),
                    note='Exploratory selection on the reused holdout; no Kaggle score.')
    save_json(comparison, OUT / 'comparison.json')
    save_json(selected, OUT / 'selection.json')
    np.savez_compressed(OUT / 'validation_ensemble.npz', ids=reference_ids, targets=labels,
                        probabilities=fused[selected['name']], class_names=np.asarray(CLASSES))
    write_report(models, comparison, selected)
    print(f"SELECTED {selected['name']} accuracy={selected['accuracy']:.6f}", flush=True)
    if args.validation_only:
        save_json(dict(stage='validation_complete', selection=selected), OUT / 'status.json')
        return
    test_x, _, test_ids = load_arrays(rows, DEFAULT_DATA, 'test', 'outputs/data')
    test_ids = np.asarray(test_ids)
    member_probs, checkpoints = [], []
    for name, tta in zip(selected['members'], selected['tta']):
        config = dict(models[name]['config'], epochs=models[name]['metrics']['best_epoch'], final_training=True)
        output = OUT / ('final_' + name)
        save_json(dict(stage='full_training', current_run=name, selection=selected), OUT / 'status.json')
        train_stage(output, config, args.device)
        checkpoint = output / 'final.pt'
        model, restored, state = load_checkpoint(checkpoint, args.device)
        probs = predict_probabilities(model, test_x, restored['model'], args.device, restored['batch_size'], tta,
                                      image_size=restored['preprocessing']['size'])
        np.savez_compressed(output / 'test_probabilities.npz', ids=test_ids, probabilities=probs,
                            class_names=np.asarray(CLASSES))
        member_probs.append(probs)
        checkpoints.append(dict(run=name, checkpoint=str(checkpoint), epoch=state['epoch'],
                                sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    probs = fuse_probabilities(member_probs, selected['weights'], selected['temperatures'])
    np.savez_compressed(OUT / 'test_probabilities.npz', ids=test_ids, probabilities=probs,
                        class_names=np.asarray(CLASSES))
    csv_path = ROOT / 'submission_optimized.csv'
    result = write_submission(DEFAULT_TEMPLATE, dict(zip(test_ids, probs.argmax(1))), csv_path)
    if result['rows'] != 7065:
        raise ValueError('Expected 7065 competition rows')
    result.update(selection=selected, checkpoints=checkpoints, submission=str(csv_path),
                  submission_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest())
    save_json(result, OUT / 'submission_validation.json')
    write_report(models, comparison, selected, result)
    save_json(dict(stage='complete', selection=selected, submission=result), OUT / 'status.json')
    print(f"OPTIMIZATION COMPLETE: {csv_path}", flush=True)


if __name__ == '__main__':
    main()

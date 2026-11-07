"""Compare MixUp and EfficientNet, train two extra seeds, then publish a local CSV."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from emotion_classifier.data import (CLASSES, DEFAULT_DATA, DEFAULT_TEMPLATE,
                                     check_split, load_arrays, read_manifest, save_json)
from emotion_classifier.ensemble import align_probabilities, fuse_probabilities
from emotion_classifier.inference import load_checkpoint, predict_probabilities, write_submission
from evaluate import run_evaluation
from model_selection import check_requested_config, probability_metrics, train_stage
from baseline_training import completed_run_is_consistent
from train import EFFICIENTNET_PRETRAINED

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/round4'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_folders():
    return {name: ROOT / 'outputs' / parent / name for parent, names in (
        ('optimization', ('convnext_tiny', 'resnet18_smoothing', 'resnet18_weight_decay')),
        ('kaggle80', ('convnext_regularized96', 'convnext_regularized128')))
            for name in names}


def candidate_entry(name, members, weights, models, labels):
    # A member can already be in the old ensemble. Combine its weights once.
    merged = {}
    for member, weight in zip(members, weights):
        merged[member] = merged.get(member, 0.0) + weight
    members, weights = list(merged), list(merged.values())
    probs = fuse_probabilities([models[item]['probabilities'] for item in members], weights)
    return dict(name=name, members=members, weights=weights, temperatures=[1.0] * len(members),
                tta=[models[item]['tta'] for item in members], **probability_metrics(probs, labels)), probs


def write_report(models, comparison, selected, seed_choice, submission):
    baseline = read_json(OUT / 'baseline.json')
    feedback_path = OUT / 'kaggle_feedback.json'
    feedback = read_json(feedback_path) if feedback_path.exists() else None
    matched_feedback = (submission is not None and feedback is not None and
                        feedback['submission_sha256'] == submission['submission_sha256'])
    score_note = (f"用户回报本轮Kaggle得分为{feedback['kaggle_score']:.3f}，上一版为{feedback['previous_user_reported_score']:.3f}。"
                  if matched_feedback else '用户报告上一版Kaggle得分0.73；本轮新提交成绩未知。')
    lines = ['# 第四轮：MixUp、EfficientNet-B2 与多种子融合', '',
             score_note + '沿用5,633张分层分组验证图像。', '',
             'MixUp仅在原ConvNeXt配置上新增alpha=0.2、使用概率0.5，其余参数保持一致。',
             'EfficientNet-B2使用本地ImageNet权重、160输入；与原ConvNeXt比较同时改变架构与输入尺寸。',
             '所有新增验证模型训练30轮。训练准确率在MixUp启用时按混合比例加权，不能与普通训练准确率直接比较。', '',
             '| 模型 | 种子 | MixUp alpha | 输入 | 最佳轮次 | 普通验证 | 翻转TTA |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, item in models.items():
        c, m = item['config'], item['metrics']
        lines.append(f"| {name} | {c['seed']} | {c.get('mixup_alpha', 0):g} | {c['preprocessing']['size']} | {m['best_epoch']} | {m['accuracy']:.2%} | {m['flip_tta_accuracy']:.2%} |")
    lines += ['', f"多种子基础配置：{seed_choice['family']}；种子42、7、2026。选择依据为该基础模型与旧融合等权组合的验证准确率，交叉熵破同分。", '',
              '| 候选 | 验证准确率 | 交叉熵 |', '|---|---:|---:|']
    for item in comparison:
        lines.append(f"| {item['name']} | {item['accuracy']:.2%} | {item['cross_entropy']:.4f} |")
    gain = selected['accuracy'] - baseline['validation_accuracy']
    lines += ['', f"选中 **{selected['name']}**，验证准确率 **{selected['accuracy']:.2%}**；旧融合为{baseline['validation_accuracy']:.2%}，变化{gain * 100:+.2f}个百分点。", '',
              '| 类别 | 旧融合召回率 | 本轮召回率 |', '|---|---:|---:|']
    old = comparison[0]
    for name in CLASSES:
        lines.append(f"| {name} | {old['per_class'][name]:.2%} | {selected['per_class'][name]:.2%} |")
    lines += ['', '| 入选成员 | 权重 | 翻转TTA |', '|---|---:|---|']
    for name, weight, use_tta in zip(selected['members'], selected['weights'], selected['tta']):
        lines.append(f"| {name} | {weight:.4%} | {'是' if use_tta else '否'} |")
    lines += ['', '验证集用于选择最佳轮次、TTA、基础配置与融合，结果属于同一验证集上的探索，未进行独立外部验证或五折交叉验证。',
              '前两个种子的暂测组合优于三种子组合，因此额外纳入前两种子等权及其以25%/50%权重加入旧融合的3个候选。该补充比较同样使用已有验证集。',
              '没有人工修改标签、使用测试标签、复制训练/测试重叠图像标签或执行Kaggle上传。']
    if submission:
        lines += ['', f"已生成submission_round4.csv，共{submission['rows']:,}行。新增入选成员使用28,165张清洗后全量数据重训，轮数取对应验证最佳轮次，余弦调度周期保持30轮；旧成员复用。", 
                  ('全量模型没有独立验证分数；本轮Kaggle成绩来自交付后的用户回报，见outputs/round4/kaggle_feedback.json。'
                   if matched_feedback else '全量模型没有独立验证分数；新版Kaggle成绩只能由实际上传测得。'), '',
                  '执行：`D:/Anaconda3/envs/pytorch/python.exe baseline.py`；核验：`D:/Anaconda3/envs/pytorch/python.exe verify_round4.py`。']
    (ROOT / 'round4_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from task2')
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    previous = read_json(ROOT / 'outputs/kaggle80/selection.json')
    if any(value != 1.0 for value in previous['temperatures']):
        raise ValueError('This round expects the saved baseline at temperature 1')
    signature = digest(ROOT / 'outputs/data/manifest.csv')
    assert signature == previous['data_signature']
    if not (OUT / 'baseline.json').exists():
        save_json(dict(data_signature=signature, validation_accuracy=previous['accuracy'],
                       user_reported_current_kaggle_accuracy=0.73,
                       original_submissions={path: digest(ROOT / path) for path in (
                           'submission.csv', 'submission_optimized.csv', 'submission_kaggle80_candidate.csv',
                           str(DEFAULT_TEMPLATE))}), OUT / 'baseline.json')
    baseline = read_json(OUT / 'baseline.json')
    assert baseline['data_signature'] == signature
    rows = read_manifest(ROOT / 'outputs/data/manifest.csv')
    check_split(rows)
    _, labels, ids = load_arrays(rows, DEFAULT_DATA, 'val', ROOT / 'outputs/data')
    ids = np.asarray(ids)
    models = {}

    def load_source(name, folder, requested=None, fixed_tta=None):
        save_json(dict(stage='validation', current_run=name), OUT / 'status.json')
        if requested is not None:
            train_stage(folder, requested, 'cuda')
        config = read_json(folder / 'config.json')
        assert config['data_signature'] == signature and not config['final_training']
        assert completed_run_is_consistent(folder, config['epochs'], False)
        if requested is not None:
            metrics = run_evaluation(folder / 'best.pt')
        else:
            metrics = read_json(folder / 'evaluation/metrics.json')
        use_tta = metrics['selected_tta'] if fixed_tta is None else fixed_tta
        with np.load(folder / 'evaluation/validation_probabilities.npz') as data:
            assert np.array_equal(data['class_names'], np.asarray(CLASSES))
            probs = align_probabilities(ids, data['ids'], data['flip_tta_probabilities' if use_tta else 'probabilities'], labels, data['targets'])
        assert probability_metrics(probs, labels)['accuracy'] == metrics['flip_tta_accuracy' if use_tta else 'accuracy']
        models[name] = dict(folder=str(folder), config=config, metrics=metrics, tta=bool(use_tta), probabilities=probs)
        gc.collect()
        torch.cuda.empty_cache()

    for name, folder in source_folders().items():
        load_source(name, folder, fixed_tta=previous['tta'][previous['members'].index(name)])
    base = models['convnext_tiny']['config']
    requests = {
        'convnext_mixup_seed42': dict(base, mixup_alpha=0.2, mixup_probability=0.5),
        'efficientnet_b2_seed42': dict(base, model='efficientnet_b2', image_size=160,
                                     pretrained=EFFICIENTNET_PRETRAINED, mixup_alpha=0.0, mixup_probability=0.5)}
    for name, config in requests.items():
        load_source(name, OUT / name, config)
    comparison, fused_arrays = [], {}

    def add(name, members, weights):
        item, probs = candidate_entry(name, members, weights, models, labels)
        comparison.append(item)
        fused_arrays[name] = probs
        print(f"{name}: accuracy={item['accuracy']:.6f} cross_entropy={item['cross_entropy']:.4f}", flush=True)
        return item

    def plus_previous(name, extra_members, fraction):
        return add(name, previous['members'] + list(extra_members),
                   [w * (1 - fraction) for w in previous['weights']] + [fraction / len(extra_members)] * len(extra_members))

    add('previous_ensemble', previous['members'], previous['weights'])
    assert comparison[0]['accuracy'] == baseline['validation_accuracy']
    base_names = ['convnext_tiny', *requests]
    seed_scores = []
    for name in base_names:
        add('single:' + name, [name], [1.0])
        seed_scores.append(plus_previous('previous_plus_' + name + '_half', [name], 0.5))
    add('new_pair_equal', list(requests), [0.5, 0.5])
    plus_previous('previous_plus_new_pair_equal', list(requests), 2 / 3)
    preliminary = max(comparison, key=lambda item: (item['accuracy'], -item['cross_entropy']))
    save_json(comparison, OUT / 'preliminary_comparison.json')
    family = max(seed_scores, key=lambda item: (item['accuracy'], -item['cross_entropy']))['name']
    family = family.removeprefix('previous_plus_').removesuffix('_half')
    seeds = [family]
    choice = dict(family=family, seeds=[42, 7, 2026], selection_basis='previous_plus_single_half',
                  preliminary_winner=preliminary['name'])
    save_json(choice, OUT / 'seed_choice.json')
    for seed in (7, 2026):
        name = family.removesuffix('_seed42') + f'_seed{seed}'
        config = dict(models[family]['config'], seed=seed)
        load_source(name, OUT / name, config)
        seeds.append(name)
        add('single:' + name, [name], [1.0])
    add('three_seeds_equal', seeds, [1 / 3] * 3)
    for fraction in (0.25, 0.5):
        plus_previous(f'previous_plus_three_seeds_{fraction:g}', seeds, fraction)
    # Interim diagnostics showed the third seed can dilute the first two.
    add('first_two_seeds_equal', seeds[:2], [0.5, 0.5])
    for fraction in (0.25, 0.5):
        plus_previous(f'previous_plus_first_two_seeds_{fraction:g}', seeds[:2], fraction)
    # Replace only the seed42 member in the preliminary winner with its seed mean.
    members, weights = [], []
    for name, weight in zip(preliminary['members'], preliminary['weights']):
        members += seeds if name == family else [name]
        weights += [weight / 3] * 3 if name == family else [weight]
    add('preliminary_winner_seed_average', members, weights)
    selected = dict(max(comparison, key=lambda item: (item['accuracy'], -item['cross_entropy'])),
                    samples=len(labels), data_signature=signature, previous_holdout_accuracy=baseline['validation_accuracy'],
                    user_reported_current_kaggle_accuracy=0.73, kaggle_score=None,
                    note='Exploratory selection on the reused holdout; new Kaggle score is unknown.')
    save_json(comparison, OUT / 'comparison.json')
    save_json(selected, OUT / 'selection.json')
    save_json({name: {key: value for key, value in item.items() if key != 'probabilities'}
               for name, item in models.items()}, OUT / 'models.json')
    np.savez_compressed(OUT / 'validation_ensemble.npz', ids=ids, targets=labels,
                        probabilities=fused_arrays[selected['name']], class_names=np.asarray(CLASSES))
    print('SELECTED', selected['name'], selected['accuracy'], flush=True)
    write_report(models, comparison, selected, choice, None)
    if args.validation_only:
        save_json(dict(stage='validation_complete', selection=selected), OUT / 'status.json')
        return
    test_x, _, test_ids = load_arrays(rows, DEFAULT_DATA, 'test', ROOT / 'outputs/data')
    test_ids = np.asarray(test_ids)
    old_artifacts = {item['run']: item for item in read_json(ROOT / 'outputs/kaggle80/submission_validation.json')['checkpoints']}
    arrays, checkpoints = [], []
    for name, use_tta in zip(selected['members'], selected['tta']):
        source = models[name]
        config = dict(source['config'], epochs=source['metrics']['best_epoch'], final_training=True)
        save_json(dict(stage='full_training', current_run=name, selection=selected), OUT / 'status.json')
        if name in old_artifacts:
            item = dict(old_artifacts[name])
            path = Path(item['checkpoint'])
            assert item['tta'] == use_tta and digest(path) == item['sha256']
            check_requested_config(read_json(path.parent / 'config.json'), config)
            assert completed_run_is_consistent(path.parent, config['epochs'], True)
            with np.load(item['probabilities']) as data:
                assert np.array_equal(data['class_names'], np.asarray(CLASSES))
                probs = align_probabilities(test_ids, data['ids'], data['probabilities'])
        else:
            folder = OUT / ('final_' + name)
            train_stage(folder, config, 'cuda')
            path = folder / 'final.pt'
            model, restored, state = load_checkpoint(path, 'cuda')
            probs = predict_probabilities(model, test_x, restored['model'], 'cuda', restored['batch_size'],
                                          use_tta, image_size=restored['preprocessing']['size'])
            probability_path = folder / 'test_probabilities.npz'
            np.savez_compressed(probability_path, ids=test_ids, probabilities=probs, class_names=np.asarray(CLASSES))
            item = dict(run=name, checkpoint=str(path), epoch=state['epoch'], sha256=digest(path),
                        probabilities=str(probability_path), tta=use_tta)
            del model, state
            gc.collect()
            torch.cuda.empty_cache()
        arrays.append(probs)
        checkpoints.append(item)
    fused = fuse_probabilities(arrays, selected['weights'], selected['temperatures'])
    np.savez_compressed(OUT / 'test_probabilities.npz', ids=test_ids, probabilities=fused, class_names=np.asarray(CLASSES))
    csv_path = ROOT / 'submission_round4.csv'
    result = write_submission(DEFAULT_TEMPLATE, dict(zip(test_ids, fused.argmax(1))), csv_path)
    assert result['rows'] == 7065
    result.update(selection=selected, checkpoints=checkpoints, submission=str(csv_path),
                  submission_sha256=digest(csv_path), kaggle_score=None)
    for path, expected in baseline['original_submissions'].items():
        assert digest(ROOT / path) == expected
    save_json(result, OUT / 'submission_validation.json')
    save_json(dict(stage='candidate_ready', selection=selected, submission=result, kaggle_score=None), OUT / 'status.json')
    write_report(models, comparison, selected, choice, result)
    print('CANDIDATE READY', csv_path, flush=True)


if __name__ == '__main__':
    main()

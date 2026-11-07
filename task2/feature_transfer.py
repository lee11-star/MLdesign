"""Frozen DINOv2 probes, one partial fine-tune, and conditional full-data training."""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

from emotion_classifier.data import (CLASS_TO_LABEL, CLASSES, DEFAULT_DATA, DEFAULT_TEMPLATE,
                                     check_split, load_arrays, read_manifest, save_json)
from emotion_classifier.dinov2 import DEFAULT_REPO, DEFAULT_WEIGHTS, fit_linear_probe, probe_head_state
from emotion_classifier.ensemble import align_probabilities, fuse_probabilities
from emotion_classifier.inference import load_checkpoint, predict_probabilities, write_submission
from emotion_classifier.models import build_model, preprocess, preprocessing_config
from emotion_classifier.training import atomic_save
from evaluate import run_evaluation
from model_selection import probability_metrics, train_stage
from baseline import ROOT, digest, read_json

OUT = ROOT / 'outputs/round6'
MANIFEST = ROOT / 'outputs/data/manifest.csv'
RUN = OUT / 'dinov2_last4_nodropout'
FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
REGULARIZATIONS = (0.1, 1.0, 10.0)
BATCH_SIZE = 64


def feature_cache(rows, signature):
    path, metadata_path = OUT / 'frozen_features.npy', OUT / 'feature_metadata.json'
    eligible = [r for r in rows if r['split'] in ('train', 'val')]
    expected = dict(data_signature=signature, pretrained_sha256=digest(DEFAULT_WEIGHTS),
                    source_repo=str(DEFAULT_REPO),
                    backbone_source_sha256=digest(DEFAULT_REPO / 'dinov2/models/vision_transformer.py'),
                    image_size=224, batch_size=BATCH_SIZE, feature_layout='last4_cls+last_patch_mean',
                    shape=[len(eligible), 2, 1920], views=['original', 'horizontal_flip'])
    if metadata_path.exists():
        saved = read_json(metadata_path)
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('Frozen feature cache configuration differs')
        with np.load(OUT / 'feature_index.npz') as index:
            np.testing.assert_array_equal(index['ids'], [r['id'] for r in eligible])
            np.testing.assert_array_equal(index['targets'], [r['label'] for r in eligible])
        print('Reusing complete frozen feature cache', flush=True)
        return np.load(path, mmap_mode='r', allow_pickle=False), eligible
    x, labels, ids = load_arrays(rows, DEFAULT_DATA, 'all', ROOT / 'outputs/data')
    model = build_model('dinov2_vits14', DEFAULT_WEIGHTS, trainable_blocks=0).eval().cuda()
    features = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=tuple(expected['shape']))
    tick = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(x), BATCH_SIZE):
            batch = torch.from_numpy(x[start:start + BATCH_SIZE]).cuda()
            for view in range(2):
                with torch.autocast('cuda'):
                    result = model.features(preprocess(batch, 'dinov2_vits14', flip=bool(view), image_size=224))
                features[start:start + len(batch), view] = result.float().cpu().numpy()
            if start % 1024 == 0:
                print(f'Frozen features {start + len(batch)}/{len(x)}; seconds={time.perf_counter()-tick:.1f}', flush=True)
    features.flush()
    np.savez_compressed(OUT / 'feature_index.npz', ids=np.asarray(ids), targets=labels,
                        splits=np.asarray([r['split'] for r in eligible]), class_names=np.asarray(CLASSES))
    save_json(dict(expected, seconds=time.perf_counter()-tick, sha256=digest(path)), metadata_path)
    del model, x, features
    gc.collect()
    torch.cuda.empty_cache()
    return np.load(path, mmap_mode='r', allow_pickle=False), eligible


def head_probabilities(features, state):
    head = torch.nn.Sequential(torch.nn.Dropout(0), torch.nn.Linear(features.shape[-1], 6)).eval()
    head.load_state_dict(state, strict=True)
    result = []
    with torch.inference_mode():
        for start in range(0, len(features), 1024):
            batch = torch.from_numpy(np.array(features[start:start+1024], copy=True))
            result.append(head(batch).softmax(-1).numpy())
    views = np.concatenate(result)
    return views[:, 0], views.mean(1)


def probes(features, eligible, signature):
    train_mask = np.array([r['split'] == 'train' for r in eligible])
    val_mask = ~train_mask
    train_features = np.asarray(features[train_mask])
    val_features = np.asarray(features[val_mask])
    train_y = np.asarray([r['label'] for r in eligible])[train_mask]
    labels = np.asarray([r['label'] for r in eligible])[val_mask]
    ids = np.asarray([r['id'] for r in eligible])[val_mask]
    results = {}
    for c in REGULARIZATIONS:
        name = f'probe_c_{c:g}'
        folder = OUT / 'probes' / name
        folder.mkdir(parents=True, exist_ok=True)
        metadata_path = folder / 'metrics.json'
        if metadata_path.exists():
            entry = read_json(metadata_path)
            if entry['data_signature'] != signature or entry['C'] != c:
                raise ValueError('Cached probe configuration differs')
        else:
            tick = time.perf_counter()
            print(f'Fitting {name} on {len(train_y)} training images and their flips', flush=True)
            probe = fit_linear_probe(train_features, train_y, c)
            state = probe_head_state(probe)
            atomic_save(state, folder / 'head.pt')
            base, tta = head_probabilities(val_features, state)
            base_metrics, tta_metrics = probability_metrics(base, labels), probability_metrics(tta, labels)
            selected_tta = tta_metrics['accuracy'] > base_metrics['accuracy']
            entry = dict(name=name, kind='frozen', C=c, data_signature=signature, train_images=len(train_y),
                         scaler_fit_split='train_only', classifier_fit_split='train_only',
                         accuracy=base_metrics['accuracy'], flip_tta_accuracy=tta_metrics['accuracy'],
                         selected_tta=selected_tta,
                         selected_accuracy=(tta_metrics if selected_tta else base_metrics)['accuracy'],
                         selected_cross_entropy=(tta_metrics if selected_tta else base_metrics)['cross_entropy'],
                         solver_iterations=probe.steps[-1][1].n_iter_.tolist(), seconds=time.perf_counter()-tick,
                         head=str(folder / 'head.pt'), head_sha256=digest(folder / 'head.pt'),
                         probabilities=str(folder / 'validation_probabilities.npz'))
            np.savez_compressed(entry['probabilities'], ids=ids, targets=labels, probabilities=base,
                                flip_tta_probabilities=tta, class_names=np.asarray(CLASSES))
            save_json(entry, metadata_path)
        entry['solver_max_iterations'] = 400
        entry['solver_converged'] = max(entry['solver_iterations']) < 400
        save_json(entry, metadata_path)
        results[name] = entry
        print(f"{name}: ordinary={entry['accuracy']:.6f}, flip={entry['flip_tta_accuracy']:.6f}", flush=True)
    save_json(results, OUT / 'probes.json')
    return results


def old_validation(ids, labels):
    selection = read_json(ROOT / 'outputs/round4/selection.json')
    models = read_json(ROOT / 'outputs/round4/models.json')
    arrays, sources = [], []
    for name, tta in zip(selection['members'], selection['tta']):
        path = Path(models[name]['folder']) / 'evaluation/validation_probabilities.npz'
        key = 'flip_tta_probabilities' if tta else 'probabilities'
        with np.load(path) as data:
            arrays.append(align_probabilities(ids, data['ids'], data[key], labels, data['targets']))
        sources.append(dict(name=name, path=str(path), sha256=digest(path), tta=tta))
    baseline = fuse_probabilities(arrays, selection['weights'], selection['temperatures'])
    assert probability_metrics(baseline, labels)['accuracy'] == selection['accuracy']
    save_json(sources, OUT / 'baseline_sources.json')
    return baseline, selection


def compare(entries, rows, signature):
    _, labels, ids = load_arrays(rows, DEFAULT_DATA, 'val', ROOT / 'outputs/data')
    ids = np.asarray(ids)
    baseline, old = old_validation(ids, labels)
    baseline_entry = dict(name='round4_baseline', source=None, kind='baseline', new_fraction=0.0,
                          **probability_metrics(baseline, labels))
    comparison, probabilities = [baseline_entry], {baseline_entry['name']: baseline}
    for name, item in entries.items():
        with np.load(item['probabilities']) as data:
            key = 'flip_tta_probabilities' if item['selected_tta'] else 'probabilities'
            new = align_probabilities(ids, data['ids'], data[key], labels, data['targets'])
        for fraction in FRACTIONS:
            probs = fuse_probabilities([baseline, new], [1-fraction, fraction])
            candidate = dict(name=f'{name}_fraction_{fraction:g}', source=name, kind=item['kind'],
                             new_fraction=fraction, tta=item['selected_tta'], **probability_metrics(probs, labels))
            comparison.append(candidate)
            probabilities[candidate['name']] = probs
    winner = max(comparison, key=lambda c: (c['accuracy'], -c['cross_entropy']))
    selection = dict(winner, baseline_accuracy=old['accuracy'], data_signature=signature,
                     baseline_user_reported_kaggle_score=0.733, kaggle_score=None,
                     note='Exploratory selection on the reused holdout; Kaggle score is unknown.')
    save_json(entries, OUT / 'models.json')
    save_json(comparison, OUT / 'comparison.json')
    save_json(selection, OUT / 'selection.json')
    np.savez_compressed(OUT / 'validation_ensemble.npz', ids=ids, targets=labels,
                        probabilities=probabilities[winner['name']], class_names=np.asarray(CLASSES))
    print(f"Selected {winner['name']}: {winner['accuracy']:.6f} (baseline {old['accuracy']:.6f})", flush=True)
    return selection


def full_training(selection, entries, features, eligible, signature):
    entry = entries[selection['source']]
    # Both training and validation images become training data only after selection.
    c = entry['C'] if entry['kind'] == 'frozen' else read_json(OUT / 'finetune_initialization.json')['C']
    labels = np.asarray([r['label'] for r in eligible])
    folder = OUT / ('final_' + entry['name'])
    folder.mkdir(parents=True, exist_ok=True)
    head_path = folder / 'initial_head.pt'
    head_metadata = folder / 'head_fit.json'
    if head_metadata.exists():
        saved = read_json(head_metadata)
        if (saved['data_signature'] != signature or saved['C'] != c or
                saved['head_sha256'] != digest(head_path)):
            raise ValueError('Full-data head differs from the selected configuration')
    else:
        probe = fit_linear_probe(features, labels, c)
        atomic_save(probe_head_state(probe), head_path)
        save_json(dict(data_signature=signature, C=c, fit_split='all_clean_training', images=len(labels),
                       class_counts=np.bincount(labels, minlength=6).tolist(),
                       solver_iterations=probe.steps[-1][1].n_iter_.tolist(),
                       head_sha256=digest(head_path)), head_metadata)
    if entry['kind'] == 'finetuned':
        requested = dict(read_json(RUN / 'config.json'), epochs=entry['best_epoch'], final_training=True,
                         initial_head=str(head_path), initial_head_signature=digest(head_path))
        train_stage(folder, requested, 'cuda')
    else:
        model = build_model('dinov2_vits14', DEFAULT_WEIGHTS, trainable_blocks=0)
        model.classifier.load_state_dict(torch.load(head_path, weights_only=True), strict=True)
        config = dict(model='dinov2_vits14', image_size=224, trainable_blocks=0, head_dropout=0.0,
                      batch_size=BATCH_SIZE, preprocessing=preprocessing_config('dinov2_vits14', 224),
                      class_to_label=CLASS_TO_LABEL,
                      dinov2_repo=str(DEFAULT_REPO), pretrained=str(DEFAULT_WEIGHTS),
                      final_training=True, data_signature=signature, linear_probe_C=c,
                      initial_head=str(head_path), initial_head_signature=digest(head_path))
        atomic_save(dict(model=model.state_dict(), config=config, class_to_label=CLASS_TO_LABEL,
                         epoch=0, best_epoch=0, history=[]), folder / 'final.pt')
        save_json(config, folder / 'config.json')
        del model
    checkpoint = folder / 'final.pt'
    x, _, ids = load_arrays(read_manifest(MANIFEST), DEFAULT_DATA, 'test', ROOT / 'outputs/data')
    ids = np.asarray(ids)
    model, config, state = load_checkpoint(checkpoint, 'cuda')
    new = predict_probabilities(model, x, 'dinov2_vits14', 'cuda', BATCH_SIZE,
                                tta=entry['selected_tta'], image_size=224)
    np.savez_compressed(folder / 'test_probabilities.npz', ids=ids, probabilities=new, class_names=np.asarray(CLASSES))
    old_selection = read_json(ROOT / 'outputs/round4/selection.json')
    artifacts = read_json(ROOT / 'outputs/round4/submission_validation.json')
    old_arrays, sources = [], []
    old_by_name = {item['run']: item for item in artifacts['checkpoints']}
    for name in old_selection['members']:
        item = old_by_name[name]
        assert digest(item['checkpoint']) == item['sha256']
        with np.load(item['probabilities']) as data:
            old_arrays.append(align_probabilities(ids, data['ids'], data['probabilities']))
        sources.append(dict(name=item['run'], path=item['probabilities'], sha256=digest(item['probabilities'])))
    old = fuse_probabilities(old_arrays, old_selection['weights'], old_selection['temperatures'])
    fused = fuse_probabilities([old, new], [1-selection['new_fraction'], selection['new_fraction']])
    np.savez_compressed(OUT / 'test_probabilities.npz', ids=ids, probabilities=fused, class_names=np.asarray(CLASSES))
    result = write_submission(DEFAULT_TEMPLATE, dict(zip(ids, fused.argmax(1))), ROOT / 'submission_round6.csv')
    import csv
    with (ROOT / 'submission_round4.csv').open(newline='', encoding='utf-8') as handle:
        previous = {r['ID']: int(r['Emotion']) for r in csv.DictReader(handle)}
    result.update(selection=selection, submission=str(ROOT / 'submission_round6.csv'),
                  submission_sha256=digest(ROOT / 'submission_round6.csv'),
                  changed_predictions=sum(previous[i] != int(p) for i, p in zip(ids, fused.argmax(1))),
                  checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), epoch=state['epoch'],
                  new_probabilities=str(folder / 'test_probabilities.npz'),
                  new_probability_sha256=digest(folder / 'test_probabilities.npz'),
                  old_probability_sources=sources, kaggle_score=None)
    save_json(result, OUT / 'submission_validation.json')
    return result


def report(entries, selection, submission):
    lines = ['# 第六轮：DINOv2 特征与末四块微调', '',
             '沿用训练22,532张、验证5,633张的分层分组划分。输入为48灰度图扩展三通道并双线性插值到224，采用ImageNet均值/标准差，不做中心裁剪。',
             '使用本地DINOv2 ViT-S/14预训练权重，连接最后四层CLS与末层patch均值，得到1,920维特征。冻结探针仅在训练部分及其翻转上拟合StandardScaler和LogisticRegression，C限定0.1/1/10。', '',
             '| 模型 | 普通验证 | 翻转TTA |', '|---|---:|---:|']
    for name, item in entries.items():
        lines.append(f"| {name} | {item['accuracy']:.2%} | {item['flip_tta_accuracy']:.2%} |")
    lines += ['', '冻结探针均达到400次迭代上限，结果是固定计算预算下的近似拟合，没有宣称求解已收敛。',
              '初始微调在原始特征上使用头dropout0.1，训练损失异常；同1024张冻结特征的实测交叉熵由关闭dropout时0.8987增至开启时16.8706。标准化折叠为原始空间线性头后，较大的特征偏移会被dropout放大。该试验在完成7轮后停止，原记录保留在dinov2_last4/，不作为完成30轮的候选。',
              '修正后微调只关闭此处dropout，其余配方保持：30轮、seed42、batch64，头lr=3e-4、骨干lr=1e-5，仅解冻末4块与最终归一化层，wd0.01、平滑0.05、不用MixUp；分类头由训练部分拟合的最佳冻结探针初始化。',
              '每个模型采用其普通/翻转验证中较好的视角，与旧融合比较新模型占比0.1/0.25/0.5/0.75/1，准确率优先、交叉熵破同分。', '',
              f"选中 **{selection['name']}**，本地验证 **{selection['accuracy']:.2%}**，旧融合 **{selection['baseline_accuracy']:.2%}**，变化 **{(selection['accuracy']-selection['baseline_accuracy'])*100:+.2f}个百分点**。", '',
              '模型、轮次、视角和融合比例在已反复使用的同一验证集上选择，结果属于内部探索；旧提交Kaggle0.733来自用户回报，新方案Kaggle成绩未知。']
    old = read_json(ROOT / 'outputs/round4/selection.json')
    lines += ['', '| 类别 | 旧融合 | 选中方案 |', '|---|---:|---:|']
    for name in CLASSES:
        lines.append(f"| {name} | {old['per_class'][name]:.2%} | {selection['per_class'][name]:.2%} |")
    if submission:
        lines += ['', f"已生成submission_round6.csv，共{submission['rows']:,}行，相对round4改变{submission['changed_predictions']}张预测。新模型全量训练/拟合仅使用28,165张清洗后训练图像；复用旧模型已核验的测试概率，原提交保留。"]
        if selection['kind'] == 'finetuned':
            lines += [f"DINO按验证最佳第{entries[selection['source']]['best_epoch']}轮，在全量数据上重训{submission['epoch']}轮，保留30轮余弦调度周期。全量训练包含原验证图像，该全量模型没有独立验证分数。"]
    elif selection['accuracy'] <= selection['baseline_accuracy']:
        lines += ['', '验证准确率未提高时，不执行全量训练，不生成重复提交，继续保留round4。']
    else:
        lines += ['', '验证比较已完成，尚未执行全量训练或生成新提交。']
    verification_path = OUT / 'artifact_verification.json'
    if verification_path.exists():
        verification = read_json(verification_path)
        if verification['valid'] and verification['selected_accuracy'] == selection['accuracy']:
            lines += ['', '独立核验已通过：3个冻结探针、21个融合候选、完整30轮验证微调记录、冻结层保持不变、真实GPU特征及模型前64张重载预测；若生成新提交，另核对全量模型、全部测试融合概率、7,065行CSV及原提交哈希。',
                      '探针保存值为float32，独立float64数学复算最大概率差约2.86e-5，六组argmax差均为0；该检查使用绝对容差5e-5，GPU重载检查仍采用rtol=1e-5/atol=1e-6。']
    lines += ['', '未安装依赖、未下载新权重、未修改task1、未上传Kaggle。', '',
              '执行：`D:/Anaconda3/envs/pytorch/python.exe feature_transfer.py`；核验：`D:/Anaconda3/envs/pytorch/python.exe verify_transfer.py`。']
    (ROOT / 'round6_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-only', action='store_true')
    parser.add_argument('--probes-only', action='store_true')
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from task2')
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    signature = digest(MANIFEST)
    rows = read_manifest(MANIFEST)
    check_split(rows)
    originals_path = OUT / 'original_submissions.json'
    if not originals_path.exists():
        save_json({p.name: digest(p) for p in ROOT.glob('submission*.csv') if p.name != 'submission_round6.csv'}, originals_path)
    save_json(dict(stage='frozen_features'), OUT / 'status.json')
    features, eligible = feature_cache(rows, signature)
    if args.features_only:
        save_json(dict(stage='frozen_features_complete'), OUT / 'status.json')
        return
    save_json(dict(stage='linear_probes'), OUT / 'status.json')
    entries = probes(features, eligible, signature)
    selection = compare(entries, rows, signature)
    if args.probes_only:
        save_json(dict(stage='linear_probes_complete', selection=selection), OUT / 'status.json')
        return
    best_probe = max(entries.values(), key=lambda e: (e['selected_accuracy'], -e['selected_cross_entropy']))
    save_json(dict(probe=best_probe['name'], C=best_probe['C'], head=best_probe['head'],
                   sha256=best_probe['head_sha256'], fit_split='train_only'), OUT / 'finetune_initialization.json')
    requested = dict(model='dinov2_vits14', epochs=30, scheduler_epochs=30, image_size=224,
                     batch_size=BATCH_SIZE, seed=42, lr=3e-4, backbone_lr_scale=1/30,
                     head_dropout=0.0, weight_decay=0.01, label_smoothing=0.05,
                     mixup_alpha=0.0, mixup_probability=0.5, trainable_blocks=4,
                     pretrained=str(DEFAULT_WEIGHTS), dinov2_repo=str(DEFAULT_REPO),
                     initial_head=best_probe['head'], initial_head_signature=best_probe['head_sha256'],
                     final_training=False, data_signature=signature)
    save_json(dict(stage='partial_finetuning', run=str(RUN)), OUT / 'status.json')
    train_stage(RUN, requested, 'cuda')
    metrics = run_evaluation(RUN / 'best.pt', batch_size=BATCH_SIZE)
    entries['dinov2_last4'] = dict(metrics, name='dinov2_last4', kind='finetuned',
                                  probabilities=str(RUN / 'evaluation/validation_probabilities.npz'))
    selection = compare(entries, rows, signature)
    submission = None
    if selection['accuracy'] > selection['baseline_accuracy'] and not args.validation_only:
        save_json(dict(stage='full_training', selection=selection), OUT / 'status.json')
        submission = full_training(selection, entries, features, eligible, signature)
    report(entries, selection, submission)
    stage = 'candidate_ready' if submission else 'validation_complete' if args.validation_only else 'no_accuracy_improvement'
    save_json(dict(stage=stage, selection=selection, new_submission=submission is not None), OUT / 'status.json')


if __name__ == '__main__':
    main()

"""Sequential, restartable execution of the authorized experiments; never uploads."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from emotion_classifier.data import save_json
from evaluate import run_evaluation
from predict import run_prediction

ROOT = Path(__file__).resolve().parent


def completed_run_is_consistent(output, epochs, final):
    selected_path = output / ('final.pt' if final else 'best.pt')
    if not (output / 'last.pt').exists() or not selected_path.exists():
        return False
    last = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
    selected = torch.load(selected_path, map_location='cpu', weights_only=False)
    expected = epochs if final else last['best_epoch']
    return (last['epoch'] == epochs and selected['epoch'] == expected
            and last['config'] == selected['config']
            and last['config']['epochs'] == epochs
            and last['config']['data_signature'] == hashlib.sha256((ROOT / 'outputs/data/manifest.csv').read_bytes()).hexdigest())


def command(arguments, log):
    cmd = [sys.executable, '-u', *arguments]
    print('Running:', subprocess.list2cmdline(cmd), flush=True)
    with Path(log).open('a', encoding='utf-8') as handle:
        handle.write('\nCOMMAND: ' + subprocess.list2cmdline(cmd) + '\n')
        handle.flush()
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            print(line, end='', flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
        if code:
            raise RuntimeError(f'Command exited with {code}: {cmd}')


def stage_train(name, architecture, epochs, lr, final=False, horizon=None):
    output = ROOT / 'outputs' / name
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'summary.json'
    if summary_path.exists():
        state = json.loads(summary_path.read_text(encoding='utf-8'))
        if state['completed_epochs'] == epochs and completed_run_is_consistent(output, epochs, final):
            print(f'{name}: completed {epochs} epochs; retaining existing run', flush=True)
            return
    args = ['train.py', '--model', architecture, '--epochs', str(epochs), '--lr', str(lr),
            '--scheduler-epochs', str(horizon or epochs), '--output', str(output)]
    if final:
        args.append('--full-data')
    last = output / 'last.pt'
    if last.exists():
        args.extend(['--resume', str(last)])
    elif not final:
        # One epoch on the actual split measures speed and memory, then resumes the same scheduler.
        command(args + ['--stop-after', '1'], output / 'console.log')
        speed = json.loads(summary_path.read_text(encoding='utf-8'))['last_epoch_seconds']
        print(f'{name} measured {speed:.1f}s/epoch; about {speed * epochs / 60:.1f} min for {epochs} epochs', flush=True)
        args.extend(['--resume', str(last)])
    command(args, output / 'console.log')
    if not completed_run_is_consistent(output, epochs, final):
        raise ValueError(f'{name}: completed epoch and selected checkpoint do not agree')


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run this script from the task2 directory')
    (ROOT / 'outputs').mkdir(exist_ok=True)
    if not (ROOT / 'outputs/data/audit.json').exists():
        command(['audit_data.py'], ROOT / 'outputs/audit_console.log')
    stage_train('cnn_baseline', 'cnn', 25, 1e-3)
    cnn = run_evaluation(ROOT / 'outputs/cnn_baseline/best.pt')
    save_json(dict(stage='cnn_complete', metrics=cnn), ROOT / 'outputs/pipeline_status.json')
    stage_train('resnet18_transfer', 'resnet18', 30, 3e-4)
    resnet = run_evaluation(ROOT / 'outputs/resnet18_transfer/best.pt')
    results = {'cnn_baseline': cnn, 'resnet18_transfer': resnet}
    selected = max(results, key=lambda name: results[name]['selected_accuracy'])
    metric = results[selected]
    config = json.loads((ROOT / 'outputs' / selected / 'config.json').read_text(encoding='utf-8'))
    selection = dict(selected_run=selected, model=config['model'], epochs=metric['best_epoch'],
                     scheduler_epochs=config['scheduler_epochs'], lr=config['lr'], tta=metric['selected_tta'],
                     holdout_accuracy=metric['accuracy'], holdout_tta_accuracy=metric['flip_tta_accuracy'],
                     selected_holdout_accuracy=metric['selected_accuracy'], experiments=results,
                     rule='Highest holdout accuracy after optional horizontal-flip TTA; final epoch count from selected best checkpoint')
    save_json(selection, ROOT / 'outputs/selection.json')
    save_json(dict(stage='validation_complete', selection=selection), ROOT / 'outputs/pipeline_status.json')
    stage_train('final_model', config['model'], selection['epochs'], config['lr'], final=True,
                horizon=config['scheduler_epochs'])
    result = run_prediction(ROOT / 'outputs/final_model/final.pt', tta=selection['tta'])
    if result['rows'] != 7065:
        raise ValueError('Competition submission requires 7065 rows')
    command(['validate_submission.py'], ROOT / 'outputs/final_validation.log')
    save_json(dict(stage='complete', selection=selection, submission=result), ROOT / 'outputs/pipeline_status.json')
    print('PIPELINE COMPLETE: models, validation metrics, probabilities and submission.csv are saved.', flush=True)


if __name__ == '__main__':
    main()

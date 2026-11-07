from pathlib import Path

from predict import predict
from train import parser, train_pipeline


if __name__ == '__main__':
    args = parser().parse_args()
    summary = train_pipeline(args)
    if not args.validation_only:
        predict(summary['full']['checkpoint'], args.data, str(Path(args.output).resolve().parent / 'submission.csv'), args.device, args.output)

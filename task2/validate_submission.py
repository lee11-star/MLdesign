import argparse
import json

from emotion_classifier.data import DEFAULT_TEMPLATE
from emotion_classifier.inference import validate_submission


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('submission', nargs='?', default='submission.csv')
    p.add_argument('--template', default=str(DEFAULT_TEMPLATE))
    p.add_argument('--expected-rows', type=int, default=7065)
    args = p.parse_args()
    print(json.dumps(validate_submission(args.submission, args.template, args.expected_rows), ensure_ascii=False, indent=2))

import argparse
import json
from pathlib import Path

from plant_classifier.core import build_class_mapping, discover_data, image_files, validate_submission


def main():
    parser = argparse.ArgumentParser(description="Validate final competition submission")
    parser.add_argument("submission", type=Path, nargs="?", default=Path("submission.csv"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(); paths = discover_data(args.root)
    classes = list(build_class_mapping(paths.train_dir)); test_ids = [p.name for p in image_files(paths.test_dir)]
    print(json.dumps(validate_submission(args.submission, test_ids, classes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

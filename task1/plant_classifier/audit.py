from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .core import build_class_mapping, discover_data, image_files


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray)
    bits = arr[:, 1:] > arr[:, :-1]
    return f"{int(''.join('1' if x else '0' for x in bits.flat), 2):0{hash_size*hash_size//4}x}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_groups(records: list[dict], key: str) -> list[list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record[key]].append(record)
    return [items for items in grouped.values() if len(items) > 1]


def run_audit(root: Path, output_dir: Path) -> dict:
    paths = discover_data(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = build_class_mapping(paths.train_dir)
    records: list[dict] = []
    unreadable: list[str] = []
    for split, base in (("train", paths.train_dir), ("test", paths.test_dir)):
        entries = []
        if split == "train":
            for class_name in mapping:
                entries.extend((p, class_name) for p in image_files(base / class_name))
        else:
            entries.extend((p, None) for p in image_files(base))
        for path, class_name in entries:
            try:
                with Image.open(path) as image:
                    image.load()
                    record = {
                        "split": split,
                        "class": class_name,
                        "filename": path.name,
                        "path": str(path),
                        "width": image.width,
                        "height": image.height,
                        "aspect_ratio": image.width / image.height,
                        "mode": image.mode,
                        "channels": len(image.getbands()),
                        "sha256": sha256_file(path),
                        "dhash": difference_hash(image),
                    }
                    records.append(record)
            except (OSError, UnidentifiedImageError) as exc:
                unreadable.append(f"{path}: {exc}")
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "image_inventory.csv", index=False, encoding="utf-8")
    exact_groups = _duplicate_groups(records, "sha256")
    perceptual_groups = _duplicate_groups(records, "dhash")

    train_names = [r["filename"] for r in records if r["split"] == "train"]
    test_names = [r["filename"] for r in records if r["split"] == "test"]
    template = pd.read_csv(paths.submission_template)
    template_ids = template["ID"].astype(str).tolist()
    test_set, template_set = set(test_names), set(template_ids)

    train_exact = [g for g in exact_groups if all(x["split"] == "train" for x in g)]
    cross_class_exact = [g for g in train_exact if len({x["class"] for x in g}) > 1]
    train_test_exact = [g for g in exact_groups if {x["split"] for x in g} == {"train", "test"}]
    train_test_perceptual = [g for g in perceptual_groups if {x["split"] for x in g} == {"train", "test"}]
    basename_duplicates = sorted([name for name, count in Counter(train_names + test_names).items() if count > 1])

    summary = {
        "paths": {"train": str(paths.train_dir), "test": str(paths.test_dir), "submission_template": str(paths.submission_template)},
        "classes": list(mapping),
        "class_counts": {name: len(image_files(paths.train_dir / name)) for name in mapping},
        "train_count": int((frame["split"] == "train").sum()),
        "test_count": int((frame["split"] == "test").sum()),
        "unreadable": unreadable,
        "modes": {
            split: {str(mode): int(count) for mode, count in group["mode"].value_counts().items()}
            for split, group in frame.groupby("split")
        },
        "channels": {str(k): int(v) for k, v in frame["channels"].value_counts().items()},
        "width": {k: float(v) for k, v in frame["width"].describe().items()},
        "height": {k: float(v) for k, v in frame["height"].describe().items()},
        "aspect_ratio": {k: float(v) for k, v in frame["aspect_ratio"].describe().items()},
        "basename_duplicates": basename_duplicates,
        "exact_duplicate_groups_train": len(train_exact),
        "exact_cross_class_groups_train": len(cross_class_exact),
        "exact_train_test_groups": len(train_test_exact),
        "same_dhash_train_test_groups": len(train_test_perceptual),
        "template_columns": template.columns.tolist(),
        "template_rows": len(template),
        "template_duplicate_ids": int(template["ID"].duplicated().sum()),
        "template_missing_ids": sorted(test_set - template_set),
        "template_extra_ids": sorted(template_set - test_set),
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    duplicate_payload = {
        "exact_groups": [[{k: r[k] for k in ("split", "class", "filename", "sha256")} for r in g] for g in exact_groups],
        "same_dhash_groups": [[{k: r[k] for k in ("split", "class", "filename", "dhash")} for r in g] for g in perceptual_groups],
    }
    (output_dir / "duplicate_groups.json").write_text(json.dumps(duplicate_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.figure(figsize=(9, 4.5))
    pd.Series(summary["class_counts"]).plot.bar(color="#4c956c")
    plt.ylabel("Images"); plt.title("Training class distribution"); plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=160); plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(frame["width"], bins=25, alpha=.7, label="width"); axes[0].hist(frame["height"], bins=25, alpha=.7, label="height")
    axes[0].legend(); axes[0].set_title("Image dimensions")
    axes[1].hist(frame["aspect_ratio"], bins=25, color="#f28f3b"); axes[1].set_title("Aspect ratio")
    fig.tight_layout(); fig.savefig(output_dir / "image_size_distribution.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(len(mapping), 4, figsize=(10, 2.4 * len(mapping)), squeeze=False)
    for row, class_name in enumerate(mapping):
        samples = image_files(paths.train_dir / class_name)[:4]
        for col, sample in enumerate(samples):
            with Image.open(sample) as image:
                axes[row, col].imshow(image.convert("RGB"))
            axes[row, col].axis("off"); axes[row, col].set_title(class_name if col == 0 else sample.name, fontsize=8)
    fig.tight_layout(); fig.savefig(output_dir / "class_samples.png", dpi=160); plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit plant image dataset")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audit"))
    args = parser.parse_args()
    print(json.dumps(run_audit(args.root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

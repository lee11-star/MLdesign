from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class DataPaths:
    train_dir: Path
    test_dir: Path
    submission_template: Path


def image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def build_class_mapping(train_dir: Path) -> dict[str, int]:
    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No class directories found in {train_dir}")
    return {name: i for i, name in enumerate(classes)}


def discover_data(root: Path) -> DataPaths:
    root = root.resolve()
    candidates: list[tuple[Path, Path]] = []
    for train in root.rglob("train"):
        test = train.parent / "test"
        if train.is_dir() and test.is_dir() and any(p.is_dir() for p in train.iterdir()):
            candidates.append((train, test))
    if not candidates:
        raise FileNotFoundError(f"Could not locate matching train/test directories under {root}")
    candidates.sort(key=lambda x: (len(x[0].parts), str(x[0])))
    train_dir, test_dir = candidates[0]
    test_ids = {p.name for p in image_files(test_dir)}
    templates = sorted(root.rglob("submission-for-task2.csv"), key=lambda p: (len(p.parts), str(p)))
    matching: list[Path] = []
    for path in templates:
        try:
            df = pd.read_csv(path)
            if list(df.columns) == ["ID", "Category"] and set(df["ID"].astype(str)) == test_ids:
                matching.append(path)
        except Exception:
            continue
    if not matching:
        raise FileNotFoundError("No submission template whose IDs exactly match the test images")
    return DataPaths(train_dir.resolve(), test_dir.resolve(), matching[0].resolve())


def stratified_split(labels: Sequence[int], val_size: float, seed: int) -> tuple[list[int], list[int]]:
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=seed, stratify=np.asarray(labels)
    )
    return sorted(train_idx.tolist()), sorted(val_idx.tolist())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_mapping(mapping: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"class_to_idx": mapping, "idx_to_class": {str(v): k for k, v in mapping.items()}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_submission(path: Path, expected_ids: Sequence[str], classes: Sequence[str]) -> dict[str, int]:
    df = pd.read_csv(path)
    if list(df.columns) != ["ID", "Category"]:
        raise ValueError(f"Columns must be exactly ID,Category; got {list(df.columns)}")
    if df["ID"].isna().any() or df["Category"].isna().any():
        raise ValueError("Submission contains missing values")
    if df["ID"].duplicated().any():
        raise ValueError("Submission contains duplicate IDs")
    expected = set(map(str, expected_ids))
    actual = set(df["ID"].astype(str))
    if actual != expected:
        raise ValueError(f"ID set mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    invalid = sorted(set(df["Category"].astype(str)) - set(classes))
    if invalid:
        raise ValueError(f"Invalid categories: {invalid}")
    reread = pd.read_csv(path, encoding="utf-8")
    if not reread.equals(df):
        raise ValueError("UTF-8 CSV round-trip mismatch")
    return {"rows": len(df), "unique_ids": df["ID"].nunique(), "classes_used": df["Category"].nunique()}


def merge_exact_frames(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], suffixes: tuple[str, str]) -> pd.DataFrame:
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("OOF keys must be unique in both frames")
    left_keys = set(map(tuple, left[keys].itertuples(index=False, name=None)))
    right_keys = set(map(tuple, right[keys].itertuples(index=False, name=None)))
    if left_keys != right_keys:
        raise ValueError(f"OOF key sets differ: missing_right={len(left_keys-right_keys)}, missing_left={len(right_keys-left_keys)}")
    merged = left.merge(right, on=keys, suffixes=suffixes, validate="one_to_one", how="inner", sort=False)
    if len(merged) != len(left):
        raise ValueError(f"OOF row count changed during merge: {len(left)} -> {len(merged)}")
    return merged

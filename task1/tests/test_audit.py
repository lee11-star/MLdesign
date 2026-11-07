import json
from pathlib import Path

import pandas as pd
from PIL import Image

from plant_classifier.audit import difference_hash, run_audit


def test_difference_hash_is_stable_for_identical_pixels():
    image = Image.new("RGB", (20, 20), (10, 100, 30))
    assert difference_hash(image) == difference_hash(image.copy())


def test_difference_hash_has_64_bits():
    image = Image.new("L", (8, 8), 0)
    value = difference_hash(image)
    assert len(value) == 16


def test_run_audit_writes_json_with_string_mode_keys(tmp_path: Path):
    data = tmp_path / "data"
    (data / "train" / "A").mkdir(parents=True)
    (data / "test").mkdir()
    Image.new("RGB", (12, 10), "green").save(data / "train" / "A" / "a.png")
    Image.new("RGB", (10, 12), "blue").save(data / "test" / "x.png")
    pd.DataFrame({"ID": ["x.png"], "Category": ["A"]}).to_csv(
        tmp_path / "submission-for-task2.csv", index=False
    )
    run_audit(tmp_path, tmp_path / "out")
    payload = json.loads((tmp_path / "out" / "audit_summary.json").read_text("utf-8"))
    assert payload["modes"] == {"test": {"RGB": 1}, "train": {"RGB": 1}}

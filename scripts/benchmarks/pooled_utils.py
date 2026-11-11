"""Shared IO and frozen pooled-feature loading."""

from __future__ import annotations
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import hvangle_frozen_spatial_benchmark as spatial

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ordered_records(
    splits: dict[str, list[spatial.HVRecord]],
) -> list[spatial.HVRecord]:
    return [record for split in SPLITS for record in splits[split]]


def split_indices(splits: dict[str, list[spatial.HVRecord]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    start = 0
    for split in SPLITS:
        stop = start + len(splits[split])
        result[split] = np.arange(start, stop, dtype=np.int64)
        start = stop
    return result


def load_features(config, model_id, expected_rows, feature_root=None):
    spec = config["models"][model_id]
    folder = (feature_root or ROOT / spec["cache"]) / "features" / model_id
    metadata = json.loads((folder / "pooled.json").read_text())
    feature_path = folder / "pooled.npy"
    if metadata["feature_sha256"] != sha256_file(feature_path):
        raise ValueError(f"Corrupt feature array: {feature_path}")
    locked = json.loads((ROOT / config["angle_result_manifest"]).read_text())[
        "split_ids"
    ]
    order = [name for split in SPLITS for name in locked[split]]
    if metadata["ids_sha256"] != spatial.sha256_ids(order):
        raise ValueError("Feature row order does not match the split")
    if metadata["dataset_csv_sha256"] != sha256_file(spatial.DATASET_CSV):
        raise ValueError("Dataset annotations changed since feature extraction")
    if metadata["angle_manifest_sha256"] != sha256_file(
        ROOT / config["angle_result_manifest"]
    ):
        raise ValueError("Split manifest changed since feature extraction")
    values = np.load(feature_path, mmap_mode="r")
    if values.shape != (expected_rows, spec["pooled_dim"]):
        raise ValueError(f"Unexpected feature shape: {values.shape}")
    return np.array(values, dtype=np.float32, copy=True), metadata

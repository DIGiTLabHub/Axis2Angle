#!/usr/bin/env python3
"""Frozen feature extraction and dense anatomical-axis mask training."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("hvangle_frozen_spatial_config.json")
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "controlled_cross_model_v2"
    / "medical"
    / "hvangle_spatial"
)
DATA_DIR = ROOT / "HVAngleEst" / "HVAngleEst"
DATASET_CSV = DATA_DIR / "datasets.csv"
os.environ.setdefault("TORCH_HOME", str(ROOT / "outputs" / "model_cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / "outputs" / "model_cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


LINE_NAMES = ("great_toe", "first_metatarsal", "second_metatarsal")
SPLIT_NAMES = ("train", "val", "test")
_ENCODER_FINGERPRINTS: dict[str, dict] = {}


@dataclass
class HVRecord:
    identifier: str
    image_path: str
    patient_id: int
    width: int
    height: int
    laterality: str
    lines: np.ndarray
    targets: np.ndarray
    split: str


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def encoder_fingerprint(spec: dict) -> dict:
    path = ROOT / spec["checkpoint"]
    key = sha256_json(
        {
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "model_config": spec.get("model_config"),
            "model_config_sha256": spec.get("model_config_sha256"),
        }
    )
    if key in _ENCODER_FINGERPRINTS:
        return _ENCODER_FINGERPRINTS[key]
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen-encoder checkpoint: {path}")
    observed = sha256_file(path)
    expected = str(spec["checkpoint_sha256"])
    if observed != expected:
        raise ValueError(
            f"checkpoint hash mismatch for {spec['display_name']}: {observed} != {expected}"
        )
    result = {
        "path": str(path.relative_to(ROOT)),
        "sha256": observed,
        "bytes": int(path.stat().st_size),
    }
    if "model_config" in spec:
        config_path = ROOT / spec["model_config"]
        if not config_path.is_file():
            raise FileNotFoundError(
                f"missing frozen-encoder model config: {config_path}"
            )
        config_observed = sha256_file(config_path)
        config_expected = str(spec["model_config_sha256"])
        if config_observed != config_expected:
            raise ValueError(
                f"model-config hash mismatch for {spec['display_name']}: "
                f"{config_observed} != {config_expected}"
            )
        result["model_config"] = {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": config_observed,
            "bytes": int(config_path.stat().st_size),
        }
    _ENCODER_FINGERPRINTS[key] = result
    return result


def parse_numbers(value: object, expected: int) -> np.ndarray:
    array = np.asarray([float(item) for item in str(value).split(",")], np.float32)
    if array.size != expected or not np.isfinite(array).all():
        raise ValueError(
            f"expected {expected} finite comma-separated values, got {value!r}"
        )
    return array


def letterbox_parameters(
    width: int, height: int, size: int
) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions: {width}x{height}")
    if width / height > 1.0:
        resized_width, resized_height = size, round(height / width * size)
    elif width / height < 1.0:
        resized_width, resized_height = round(width / height * size), size
    else:
        resized_width = resized_height = size
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    return resized_width, resized_height, left, top


def letterbox_image(image: Image.Image, size: int) -> Image.Image:
    contained = ImageOps.contain(image, (size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), color=0)
    canvas.paste(
        contained, ((size - contained.width) // 2, (size - contained.height) // 2)
    )
    return canvas


def preprocess_path(
    preprocess: Callable[[Image.Image], torch.Tensor], path: str
) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess(image)


def line_to_canvas(line: np.ndarray, width: int, height: int, size: int) -> np.ndarray:
    resized_width, resized_height, left, top = letterbox_parameters(width, height, size)
    result = np.asarray(line, np.float32).copy()
    result[:, 0] = left + result[:, 0] * resized_width
    result[:, 1] = top + result[:, 1] * resized_height
    return result


def load_locked_records(config: dict) -> tuple[dict[str, list[HVRecord]], dict]:
    angle_manifest_path = ROOT / config["angle_result_manifest"]
    if not angle_manifest_path.is_file():
        raise FileNotFoundError(f"missing locked angle manifest: {angle_manifest_path}")
    angle_manifest = json.loads(angle_manifest_path.read_text())
    split_ids = angle_manifest.get("split_ids", {})
    expected_sizes = {"train": 829, "val": 172, "test": 176}
    observed_sizes = {name: len(split_ids.get(name, [])) for name in SPLIT_NAMES}
    if observed_sizes != expected_sizes:
        raise ValueError(
            f"locked split sizes changed: {observed_sizes} != {expected_sizes}"
        )
    flattened = [identifier for split in SPLIT_NAMES for identifier in split_ids[split]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("locked split contains duplicate image identifiers")

    frame = pd.read_csv(DATASET_CSV)
    selected = frame[frame.filename.isin(flattened)].copy()
    counts = selected.filename.value_counts()
    if set(counts.index) != set(flattened) or not bool((counts == 1).all()):
        raise ValueError(
            "locked identifiers do not select exactly one HVAngleEst row each"
        )
    by_name = selected.set_index("filename", drop=False)
    split_records: dict[str, list[HVRecord]] = {}
    image_dimension_mismatches = []
    canonical_axis_lengths = []
    for split in SPLIT_NAMES:
        rows = []
        for identifier in split_ids[split]:
            row = by_name.loc[identifier]
            lines = np.stack(
                [parse_numbers(row[name], 4).reshape(2, 2) for name in LINE_NAMES]
            )
            if np.any(lines < 0.0) or np.any(lines > 1.0):
                raise ValueError(
                    f"out-of-range protocol-reference endpoint: {identifier}"
                )
            image_path = DATA_DIR / "images" / identifier
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                if image.size != (int(row.image_width), int(row.image_height)):
                    image_dimension_mismatches.append(
                        {
                            "identifier": identifier,
                            "csv_size": [int(row.image_width), int(row.image_height)],
                            "image_size": list(image.size),
                        }
                    )
            for class_index, line in enumerate(lines):
                canonical = line_to_canvas(
                    line,
                    int(row.image_width),
                    int(row.image_height),
                    int(config["image_size"]),
                )
                canonical_axis_lengths.append(
                    float(np.linalg.norm(canonical[1] - canonical[0]))
                )
            records = HVRecord(
                identifier=identifier,
                image_path=str(image_path),
                patient_id=int(row.patient_id),
                width=int(row.image_width),
                height=int(row.image_height),
                laterality=str(row.labels),
                lines=lines,
                targets=np.asarray([float(row.HVA), float(row.IMA)], np.float32),
                split=split,
            )
            rows.append(records)
        split_records[split] = rows

    patient_sets = {
        name: {row.patient_id for row in rows} for name, rows in split_records.items()
    }
    if (
        patient_sets["train"] & patient_sets["val"]
        or patient_sets["train"] & patient_sets["test"]
        or patient_sets["val"] & patient_sets["test"]
    ):
        raise ValueError("locked split is not patient-disjoint")
    patient_counts = {name: len(values) for name, values in patient_sets.items()}
    if patient_counts != {"train": 617, "val": 132, "test": 133}:
        raise ValueError(f"locked patient counts changed: {patient_counts}")

    audit = {
        "protocol": config["protocol_version"],
        "dataset_csv": str(DATASET_CSV.relative_to(ROOT)),
        "dataset_csv_sha256": sha256_file(DATASET_CSV),
        "angle_manifest": str(angle_manifest_path.relative_to(ROOT)),
        "angle_manifest_sha256": sha256_file(angle_manifest_path),
        "split_sizes": observed_sizes,
        "split_patient_counts": patient_counts,
        "split_id_sha256": {name: sha256_ids(split_ids[name]) for name in SPLIT_NAMES},
        "image_dimension_mismatches": image_dimension_mismatches,
        "minimum_canonical_axis_length_px": float(min(canonical_axis_lengths)),
        "split_laterality_counts": {
            name: {
                label: sum(row.laterality == label for row in rows)
                for label in sorted({record.laterality for record in rows})
            }
            for name, rows in split_records.items()
        },
        "line_source": "datasets.csv endpoint pairs, rendered as three overlapping binary channels",
    }
    return split_records, audit


def pooled_feature_cache(
    model_id: str,
    spec: dict,
    records: list[HVRecord],
    config: dict,
    output: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    cache_dir = output / "features" / model_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path, manifest_path = cache_dir / "pooled.npy", cache_dir / "pooled.json"
    expected_ids = [row.identifier for row in records]
    source = encoder_fingerprint(spec)
    config_sha256 = sha256_json(config)
    implementation_sha256 = sha256_file(Path(__file__))
    dataset_sha256 = sha256_file(DATASET_CSV)
    angle_manifest_sha256 = sha256_file(ROOT / config["angle_result_manifest"])
    if array_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("ids_sha256") == sha256_ids(expected_ids)
            and manifest.get("protocol") == config["protocol_version"]
            and manifest.get("family") == spec["family"]
            and manifest.get("config_sha256") == config_sha256
            and manifest.get("implementation_sha256") == implementation_sha256
            and manifest.get("dataset_csv_sha256") == dataset_sha256
            and manifest.get("angle_manifest_sha256") == angle_manifest_sha256
            and manifest.get("encoder_source", {}).get("sha256") == source["sha256"]
        ):
            array = np.load(array_path, mmap_mode="r")
            if tuple(array.shape) == (len(records), int(spec["pooled_dim"])):
                return array, manifest
        print(f"regenerating stale pooled feature cache: {array_path}", flush=True)

    features, encoder_parameters = extract_pooled_features(
        spec, records, int(config["batch_size"]), int(config["image_size"]), device
    )
    if features.shape != (len(records), int(spec["pooled_dim"])):
        raise ValueError(f"{model_id} pooled feature shape changed: {features.shape}")
    np.save(array_path, np.asarray(features, np.float32))
    manifest = {
        "protocol": config["protocol_version"],
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "dataset_csv_sha256": dataset_sha256,
        "angle_manifest_sha256": angle_manifest_sha256,
        "model_id": model_id,
        "family": spec["family"],
        "display_name": spec["display_name"],
        "interface": "pooled frozen interface used by the existing HVA/IMA probe",
        "shape": list(features.shape),
        "dtype": "float32",
        "ids_sha256": sha256_ids(expected_ids),
        "encoder_parameters": int(encoder_parameters),
        "encoder_source": source,
        "feature_file": str(array_path.relative_to(ROOT)),
        "feature_sha256": sha256_file(array_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return np.load(array_path, mmap_mode="r"), manifest


def extract_pooled_features(
    spec: dict,
    records: list[HVRecord],
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    """Reproduce the existing angle-probe pooled interface without network fallback."""
    adapter = build_dense_adapter(spec, device, batch_size, image_size)
    family = spec["family"]
    projection_weight = None
    encoder_parameters = adapter.parameters
    if family == "medclip_vit":
        checkpoint = torch.load(
            ROOT / spec["checkpoint"], map_location="cpu", weights_only=False
        )
        projection_weight = checkpoint["vision_model.projection_head.weight"].to(device)
        encoder_parameters += int(projection_weight.numel())
    elif family == "medimageinsight":
        encoder_parameters += int(adapter.model.image_projection.numel())
    chunks = []
    native_batch = adapter.native_batch_size
    for start in range(0, len(records), native_batch):
        batch = records[start : start + native_batch]
        images = torch.stack(
            [preprocess_path(adapter.preprocess, row.image_path) for row in batch]
        ).to(device)
        with torch.inference_mode():
            if family == "biomedclip":
                encoded = adapter.model.encode_image(images)
            elif family == "medimageinsight":
                encoded = adapter.model.encode_image(images)
            elif family == "medclip_resnet":
                encoded = adapter.model(images)
            elif family == "medclip_vit":
                values = images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images
                pooled = adapter.model(
                    pixel_values=values, return_dict=True
                ).pooler_output
                encoded = F.linear(pooled, projection_weight)
            elif family in {"dax_resnet50", "imagenet_resnet50"}:
                encoded = adapter.model(images)
            else:
                raise ValueError(f"unsupported pooled family: {family}")
            encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        chunks.append(encoded.to(device="cpu", dtype=torch.float32).numpy())
        print(
            f"[{spec['display_name']}] pooled features "
            f"{start + len(batch)}/{len(records)}",
            flush=True,
        )
    features = np.concatenate(chunks)
    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, encoder_parameters


def summarize_seed_metrics(runs: list[dict], fields: Iterable[str]) -> dict:
    summary = {}
    for field in fields:
        values = np.asarray([run["metrics"][field] for run in runs], float)
        summary[field] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        }
    return summary


def render_line_masks(
    record: HVRecord, image_size: int, line_width: float
) -> np.ndarray:
    grid_y, grid_x = np.mgrid[:image_size, :image_size]
    pixel_centers = np.stack([grid_x + 0.5, grid_y + 0.5], axis=-1)
    masks = []
    for line in record.lines:
        points = line_to_canvas(line, record.width, record.height, image_size)
        start, end = points
        vector = end - start
        length_squared = float(np.dot(vector, vector))
        if length_squared <= 0:
            raise ValueError(f"zero-length line: {record.identifier}")
        relative = pixel_centers - start
        fraction = np.clip((relative @ vector) / length_squared, 0.0, 1.0)
        closest = start + fraction[..., None] * vector
        distance = np.linalg.norm(pixel_centers - closest, axis=-1)
        masks.append(distance <= line_width / 2.0)
    return np.stack(masks).astype(np.uint8)


def line_target_cache(
    records: list[HVRecord], config: dict, output: Path
) -> tuple[np.ndarray, dict]:
    cache_dir = output / "targets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path, manifest_path = (
        cache_dir / "line_masks.npy",
        cache_dir / "line_masks.json",
    )
    image_size = int(config["image_size"])
    line_width = float(config["lines"]["canonical_line_width_px"])
    expected_shape = (len(records), 3, image_size, image_size)
    ids = [row.identifier for row in records]
    dataset_sha256 = sha256_file(DATASET_CSV)
    config_sha256 = sha256_json(config)
    implementation_sha256 = sha256_file(Path(__file__))
    if array_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        array = np.load(array_path, mmap_mode="r")
        if (
            tuple(array.shape) == expected_shape
            and manifest.get("ids_sha256") == sha256_ids(ids)
            and float(manifest.get("line_width_px", -1)) == line_width
            and manifest.get("dataset_csv_sha256") == dataset_sha256
            and manifest.get("implementation_sha256") == implementation_sha256
            and manifest.get("config_sha256") == config_sha256
        ):
            return array, manifest
        print(f"regenerating stale line-target cache: {array_path}", flush=True)
    array = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.uint8, shape=expected_shape
    )
    for index, record in enumerate(records):
        array[index] = render_line_masks(record, image_size, line_width)
    array.flush()
    positive = np.asarray(array, dtype=np.uint8).sum(axis=(0, 2, 3)).astype(np.int64)
    total = len(records) * image_size * image_size
    manifest = {
        "protocol": config["protocol_version"],
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "dataset_csv_sha256": dataset_sha256,
        "shape": list(expected_shape),
        "dtype": "uint8",
        "ids_sha256": sha256_ids(ids),
        "line_width_px": line_width,
        "renderer": "pixel-center distance to finite endpoint-defined segment <= width/2",
        "overlap_policy": "three independent binary channels; intersections are retained",
        "positive_pixels": {
            name: int(value) for name, value in zip(LINE_NAMES, positive)
        },
        "positive_fraction": {
            name: float(value / total) for name, value in zip(LINE_NAMES, positive)
        },
        "target_file": str(array_path.relative_to(ROOT)),
        "target_sha256": sha256_file(array_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return np.load(array_path, mmap_mode="r"), manifest


def line_template_baselines(
    splits: dict[str, list[HVRecord]], endpoint_masks: np.ndarray, config: dict
) -> dict:
    """Quantify image-independent coordinate priors for the line task."""
    train, test = splits["train"], splits["test"]
    test_start = len(splits["train"]) + len(splits["val"])
    truth_masks = np.asarray(endpoint_masks[test_start:], np.uint8)
    unconditional = np.median(np.stack([row.lines for row in train]), axis=0)
    laterality = {
        label: np.median(
            np.stack([row.lines for row in train if row.laterality == label]), axis=0
        )
        for label in sorted({row.laterality for row in train})
    }

    def evaluate(template_for: Callable[[HVRecord], np.ndarray]) -> dict:
        case_dice, case_cldice, case_axis = [], [], []
        class_dice, class_cldice, class_axis = [[], [], []], [[], [], []], [[], [], []]
        for index, record in enumerate(test):
            template_lines = np.asarray(template_for(record), np.float32)
            template_record = HVRecord(
                record.identifier,
                record.image_path,
                record.patient_id,
                record.width,
                record.height,
                record.laterality,
                template_lines,
                record.targets,
                record.split,
            )
            predicted_masks = render_line_masks(
                template_record,
                int(config["image_size"]),
                float(config["lines"]["canonical_line_width_px"]),
            )
            dice_values, cldice_values, axis_values = [], [], []
            for class_index in range(3):
                prediction = predicted_masks[class_index].astype(bool)
                truth = truth_masks[index, class_index].astype(bool)
                intersection = 2.0 * np.logical_and(prediction, truth).sum()
                dice = float(intersection / max(prediction.sum() + truth.sum(), 1e-12))
                predicted_skeleton = morphological_skeleton(prediction)
                true_skeleton = morphological_skeleton(truth)
                topology_precision = float(
                    np.logical_and(predicted_skeleton, truth).sum()
                    / max(predicted_skeleton.sum(), 1e-12)
                )
                topology_sensitivity = float(
                    np.logical_and(true_skeleton, prediction).sum()
                    / max(true_skeleton.sum(), 1e-12)
                )
                cldice = float(
                    2
                    * topology_precision
                    * topology_sensitivity
                    / max(topology_precision + topology_sensitivity, 1e-12)
                )
                predicted_line = template_lines[class_index] * np.asarray(
                    [record.width, record.height], np.float32
                )
                truth_line = record.lines[class_index] * np.asarray(
                    [record.width, record.height], np.float32
                )
                error = axis_error(predicted_line[1] - predicted_line[0], truth_line)
                class_dice[class_index].append(dice)
                class_cldice[class_index].append(cldice)
                class_axis[class_index].append(error)
                dice_values.append(dice)
                cldice_values.append(cldice)
                axis_values.append(error)
            case_dice.append(float(np.mean(dice_values)))
            case_cldice.append(float(np.mean(cldice_values)))
            case_axis.append(float(np.mean(axis_values)))
        return {
            "case_macro_dice": float(np.mean(case_dice)),
            "case_macro_cldice": float(np.mean(case_cldice)),
            "macro_axis_error_deg": float(np.mean(case_axis)),
            "class_case_mean_dice": {
                name: float(np.mean(class_dice[index]))
                for index, name in enumerate(LINE_NAMES)
            },
            "class_case_mean_cldice": {
                name: float(np.mean(class_cldice[index]))
                for index, name in enumerate(LINE_NAMES)
            },
            "class_axis_error_deg": {
                name: float(np.mean(class_axis[index]))
                for index, name in enumerate(LINE_NAMES)
            },
        }

    return {
        "training_coordinate_median": evaluate(lambda _: unconditional),
        "training_laterality_specific_median": evaluate(
            lambda record: laterality[record.laterality]
        ),
        "laterality_baseline_note": "metadata-assisted; it consumes the true side label",
    }


class DenseEncoderAdapter:
    def __init__(
        self,
        model: nn.Module,
        preprocess: Callable[[Image.Image], torch.Tensor],
        forward: Callable[[torch.Tensor], torch.Tensor],
        parameters: int,
        native_batch_size: int,
        interface: str,
    ):
        self.model = model
        self.preprocess = preprocess
        self.forward = forward
        self.parameters = int(parameters)
        self.native_batch_size = int(native_batch_size)
        self.interface = interface


def _freeze(model: nn.Module, device: torch.device) -> nn.Module:
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_dense_adapter(
    spec: dict, device: torch.device, batch_size: int, image_size: int
) -> DenseEncoderAdapter:
    family = spec["family"]
    if family == "biomedclip":
        import open_clip
        from open_clip import factory as open_clip_factory

        # Register the exact locally cached model definition and load the
        # separately hash-validated local checkpoint.  The hf-hub shorthand
        # is intentionally avoided because it follows a mutable ``main`` ref.
        locked_config = json.loads((ROOT / spec["model_config"]).read_text())
        local_model_name = "locked_biomedclip_vit_b16"
        open_clip_factory._MODEL_CONFIGS[local_model_name] = locked_config["model_cfg"]
        model, _, preprocess = open_clip.create_model_and_transforms(
            local_model_name,
            pretrained=str(ROOT / spec["checkpoint"]),
            pretrained_hf=False,
            image_mean=tuple(locked_config["preprocess_cfg"]["mean"]),
            image_std=tuple(locked_config["preprocess_cfg"]["std"]),
        )
        model = _freeze(model, device)

        def prepare(image: Image.Image) -> torch.Tensor:
            return preprocess(letterbox_image(image.convert("RGB"), image_size))

        def forward(images: torch.Tensor) -> torch.Tensor:
            tokens, _ = model.visual.trunk.forward_intermediates(
                images, indices=[-1], output_fmt="NLC"
            )
            tokens = tokens[:, 1:, :]
            if tokens.shape[1:] != (196, 768):
                raise ValueError(f"unexpected BiomedCLIP dense shape: {tokens.shape}")
            return tokens.transpose(1, 2).reshape(len(images), 768, 14, 14)

        parameters = sum(value.numel() for value in model.visual.parameters())
        return DenseEncoderAdapter(
            model, prepare, forward, parameters, batch_size, spec["dense_interface"]
        )

    if family == "medimageinsight":
        sys.path.insert(0, str(ROOT / "MedImageInsights"))
        from medimageinsightmodel import MedImageInsight

        checkpoint_path = ROOT / spec["checkpoint"]
        model_dir = checkpoint_path.parents[1]
        wrapper = MedImageInsight(
            str(model_dir), checkpoint_path.name, "language_model.pth"
        )
        wrapper.load_model()
        wrapper.model = _freeze(wrapper.model, device)
        image_encoder = wrapper.model.image_encoder

        def prepare(image: Image.Image) -> torch.Tensor:
            canonical = letterbox_image(image.convert("RGB"), image_size)
            buffer = io.BytesIO()
            canonical.save(buffer, format="JPEG", quality=95)
            decoded = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
            return wrapper.preprocess(decoded)

        def forward(images: torch.Tensor) -> torch.Tensor:
            spatial_size = (images.shape[2], images.shape[3])
            tokens = images
            for convolution, block in zip(image_encoder.convs, image_encoder.blocks):
                tokens, spatial_size = convolution(tokens, spatial_size)
                tokens, spatial_size = block(tokens, spatial_size)
            height, width = map(int, spatial_size)
            if tokens.shape[1:] != (height * width, 2048) or (height, width) != (
                15,
                15,
            ):
                raise ValueError(
                    f"unexpected MedImageInsight dense shape: {tokens.shape}, {spatial_size}"
                )
            return tokens.transpose(1, 2).reshape(len(images), 2048, height, width)

        parameters = sum(value.numel() for value in image_encoder.parameters())
        return DenseEncoderAdapter(
            wrapper.model,
            prepare,
            forward,
            parameters,
            min(batch_size, 4),
            spec["dense_interface"],
        )

    if family in {"medclip_resnet", "medclip_vit"}:
        sys.path.insert(0, str(ROOT / "external_models" / "MedCLIP"))
        checkpoint_path = ROOT / spec["checkpoint"]
        # The released MedCLIP file may contain CUDA storages.  Load it through
        # an explicit CPU map rather than the upstream convenience method,
        # which omits map_location and therefore fails on audit-only CPU hosts.
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        vision_state = {
            key.replace("vision_model.", ""): value
            for key, value in state_dict.items()
            if "vision_model" in key
        }
        if family == "medclip_resnet":
            import torchvision.models as tv_models

            class LocalMedCLIPResNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = tv_models.resnet50(weights=None)
                    self.model.fc = nn.Linear(
                        self.model.fc.in_features, 512, bias=False
                    )

                def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                    if pixel_values.shape[1] == 1:
                        pixel_values = pixel_values.repeat(1, 3, 1, 1)
                    return self.model(pixel_values)

            model = LocalMedCLIPResNet()
            model.load_state_dict(vision_state, strict=True)
            model = _freeze(model, device)
        else:
            # Instantiate the architecture from the hash-validated local
            # config, then load every backbone tensor from the MedCLIP file.
            # No base-model weights or network lookup are involved.
            from transformers import AutoConfig, AutoModel

            config_dir = (ROOT / spec["model_config"]).parent
            backbone_config = AutoConfig.from_pretrained(
                str(config_dir), local_files_only=True
            )
            backbone = AutoModel.from_config(backbone_config)
            backbone_state = {
                key.removeprefix("model."): value
                for key, value in vision_state.items()
                if key.startswith("model.")
            }
            backbone.load_state_dict(backbone_state, strict=True)
            model = _freeze(backbone, device)
        preprocess = T.Compose(
            [
                T.Lambda(lambda image: letterbox_image(image.convert("L"), image_size)),
                T.ToTensor(),
                T.Normalize([0.5862786], [0.2795009]),
            ]
        )

        if family == "medclip_resnet":

            def forward(images: torch.Tensor) -> torch.Tensor:
                values = images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images
                backbone = model.model
                values = backbone.relu(backbone.bn1(backbone.conv1(values)))
                values = backbone.maxpool(values)
                values = backbone.layer1(values)
                values = backbone.layer2(values)
                values = backbone.layer3(values)
                values = backbone.layer4(values)
                return values
        else:

            def forward(images: torch.Tensor) -> torch.Tensor:
                values = images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images
                tokens = model(pixel_values=values, return_dict=True).last_hidden_state
                side = int(round(math.sqrt(tokens.shape[1])))
                if side * side != tokens.shape[1]:
                    raise ValueError(
                        f"non-square MedCLIP-Swin token grid: {tokens.shape}"
                    )
                return tokens.transpose(1, 2).reshape(
                    len(images), tokens.shape[2], side, side
                )

        parameters = sum(value.numel() for value in model.parameters())
        return DenseEncoderAdapter(
            model, preprocess, forward, parameters, batch_size, spec["dense_interface"]
        )

    if family == "dax_resnet50":
        import torchvision.models as tv_models

        checkpoint = torch.load(
            ROOT / spec["checkpoint"], map_location="cpu", weights_only=False
        )
        teacher = checkpoint["teacher"]
        state = {
            key.replace("module.backbone.", ""): value
            for key, value in teacher.items()
            if key.startswith("module.backbone.")
        }
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Identity()
        model.load_state_dict(state, strict=True)
        model = _freeze(model, device)
        preprocess = T.Compose(
            [
                T.Lambda(
                    lambda image: letterbox_image(image.convert("RGB"), image_size)
                ),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        def forward(images: torch.Tensor) -> torch.Tensor:
            values = model.relu(model.bn1(model.conv1(images)))
            values = model.maxpool(values)
            values = model.layer1(values)
            values = model.layer2(values)
            values = model.layer3(values)
            return model.layer4(values)

        parameters = sum(value.numel() for value in model.parameters())
        return DenseEncoderAdapter(
            model, preprocess, forward, parameters, batch_size, spec["dense_interface"]
        )
    if family == "imagenet_resnet50":
        import torchvision.models as tv_models

        model = tv_models.resnet50(weights=None)
        state = torch.load(
            ROOT / spec["checkpoint"], map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=True)
        model.fc = nn.Identity()
        model = _freeze(model, device)
        preprocess = T.Compose(
            [
                T.Lambda(
                    lambda image: letterbox_image(image.convert("RGB"), image_size)
                ),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        def forward(images: torch.Tensor) -> torch.Tensor:
            values = model.relu(model.bn1(model.conv1(images)))
            values = model.maxpool(values)
            values = model.layer1(values)
            values = model.layer2(values)
            values = model.layer3(values)
            return model.layer4(values)

        parameters = sum(value.numel() for value in model.parameters())
        return DenseEncoderAdapter(
            model, preprocess, forward, parameters, batch_size, spec["dense_interface"]
        )
    raise ValueError(f"unsupported dense encoder family: {family}")


def dense_feature_cache(
    model_id: str,
    spec: dict,
    records: list[HVRecord],
    config: dict,
    output: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    cache_dir = output / "features" / model_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path, manifest_path = cache_dir / "dense.npy", cache_dir / "dense.json"
    expected_shape = (
        len(records),
        int(spec["dense_channels"]),
        int(spec["dense_grid"][0]),
        int(spec["dense_grid"][1]),
    )
    ids = [row.identifier for row in records]
    source = encoder_fingerprint(spec)
    config_sha256 = sha256_json(config)
    implementation_sha256 = sha256_file(Path(__file__))
    dataset_sha256 = sha256_file(DATASET_CSV)
    angle_manifest_sha256 = sha256_file(ROOT / config["angle_result_manifest"])
    if array_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        array = np.load(array_path, mmap_mode="r")
        if (
            tuple(array.shape) == expected_shape
            and manifest.get("ids_sha256") == sha256_ids(ids)
            and manifest.get("family") == spec["family"]
            and manifest.get("config_sha256") == config_sha256
            and manifest.get("implementation_sha256") == implementation_sha256
            and manifest.get("dataset_csv_sha256") == dataset_sha256
            and manifest.get("angle_manifest_sha256") == angle_manifest_sha256
            and manifest.get("encoder_source", {}).get("sha256") == source["sha256"]
        ):
            return array, manifest
        print(f"regenerating stale dense feature cache: {array_path}", flush=True)

    adapter = build_dense_adapter(
        spec, device, int(config["batch_size"]), int(config["image_size"])
    )
    array = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.float16, shape=expected_shape
    )
    batch_size = adapter.native_batch_size
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = torch.stack(
            [preprocess_path(adapter.preprocess, row.image_path) for row in batch]
        ).to(device)
        with torch.inference_mode():
            features = adapter.forward(images)
            if tuple(features.shape[1:]) != expected_shape[1:]:
                raise ValueError(
                    f"{model_id} dense feature shape {tuple(features.shape)} != {expected_shape}"
                )
            # A declared per-location normalization makes all five readouts
            # consume direction-only local feature vectors and matches the
            # existing BiomedCLIP patch cache.
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
        array[start : start + len(batch)] = features.to(
            device="cpu", dtype=torch.float16
        ).numpy()
        print(
            f"[{model_id}] dense features {start + len(batch)}/{len(records)}",
            flush=True,
        )
    array.flush()
    manifest = {
        "protocol": config["protocol_version"],
        "config_sha256": config_sha256,
        "model_id": model_id,
        "implementation_sha256": implementation_sha256,
        "dataset_csv_sha256": dataset_sha256,
        "angle_manifest_sha256": angle_manifest_sha256,
        "family": spec["family"],
        "display_name": spec.get("line_display_name", spec["display_name"]),
        "interface": adapter.interface,
        "shape": list(expected_shape),
        "dtype": "float16",
        "ids_sha256": sha256_ids(ids),
        "encoder_parameters": adapter.parameters,
        "encoder_source": source,
        "feature_file": str(array_path.relative_to(ROOT)),
        "feature_sha256": sha256_file(array_path),
        "per_location_l2_normalized": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    del adapter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.load(array_path, mmap_mode="r"), manifest


class LineFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, masks: np.ndarray, indices: np.ndarray):
        self.features = features
        self.masks = masks
        self.indices = np.asarray(indices, np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        absolute = int(self.indices[index])
        feature = torch.from_numpy(np.asarray(self.features[absolute], np.float32))
        mask = torch.from_numpy(np.asarray(self.masks[absolute], np.float32))
        return feature, mask, absolute


class ConvNormActivation(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        groups = 32 if outputs >= 32 else max(1, outputs // 4)
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outputs),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class LineDecoder(nn.Module):
    def __init__(self, input_channels: int, settings: dict):
        super().__init__()
        projection = int(settings["projection_channels"])
        groups = 32 if projection >= 32 else max(1, projection // 4)
        self.grid = int(settings["canonical_feature_grid"])
        self.project = nn.Sequential(
            nn.Conv2d(input_channels, projection, 1, bias=False),
            nn.GroupNorm(groups, projection),
            nn.GELU(),
        )
        channels = [projection, *map(int, settings["decoder_channels"])]
        self.blocks = nn.ModuleList(
            [
                ConvNormActivation(inputs, outputs)
                for inputs, outputs in zip(channels[:-1], channels[1:])
            ]
        )
        self.output = nn.Conv2d(channels[-1], 3, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(
            features, size=(self.grid, self.grid), mode="bilinear", align_corners=False
        )
        values = self.project(values)
        for block in self.blocks:
            values = F.interpolate(
                values, scale_factor=2.0, mode="bilinear", align_corners=False
            )
            values = block(values)
        return self.output(values)


def line_loss(
    logits: torch.Tensor,
    truth: torch.Tensor,
    positive_weight: torch.Tensor,
    settings: dict,
) -> tuple[torch.Tensor, dict]:
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("line decoder produced non-finite logits")
    if not bool(torch.isfinite(truth).all() and torch.isfinite(positive_weight).all()):
        raise FloatingPointError(
            "line loss received non-finite targets or class weights"
        )
    bce = F.binary_cross_entropy_with_logits(
        logits, truth, pos_weight=positive_weight.view(1, 3, 1, 1)
    )
    probability = torch.sigmoid(logits)
    intersection = (probability * truth).sum(dim=(0, 2, 3))
    denominator = probability.sum(dim=(0, 2, 3)) + truth.sum(dim=(0, 2, 3))
    soft_dice = ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    dice_loss = 1.0 - soft_dice
    total = (
        settings["bce_loss_weight"] * bce
        + settings["soft_dice_loss_weight"] * dice_loss
    )
    return total, {
        "bce": float(bce.detach()),
        "soft_dice_loss": float(dice_loss.detach()),
    }


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    current = np.asarray(mask, bool)
    skeleton = np.zeros_like(current)
    structure = ndimage.generate_binary_structure(2, 1)
    while current.any():
        opened = ndimage.binary_opening(current, structure=structure)
        skeleton |= current & ~opened
        current = ndimage.binary_erosion(current, structure=structure)
    return skeleton


def probability_axis(
    probability: np.ndarray,
    width: float,
    height: float,
    power: float,
    quantile: float,
    hard_support: np.ndarray,
    minimum_hard_pixels: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if not np.isfinite(probability).all():
        raise FloatingPointError("probability-axis fit received non-finite values")
    used_soft_fallback = int(np.asarray(hard_support, bool).sum()) < minimum_hard_pixels
    if used_soft_fallback:
        cutoff = np.quantile(probability, quantile)
        weights = np.where(probability >= cutoff, probability**power, 0.0).reshape(-1)
    else:
        weights = np.asarray(hard_support, np.float64).reshape(-1)
    map_height, map_width = probability.shape
    grid_y, grid_x = np.mgrid[:map_height, :map_width]
    coordinates = np.stack(
        [
            (grid_x + 0.5) / map_width * width,
            (grid_y + 0.5) / map_height * height,
        ],
        axis=-1,
    ).reshape(-1, 2)
    total = max(float(weights.sum()), 1e-12)
    center = (coordinates * weights[:, None]).sum(0) / total
    centered = coordinates - center
    covariance = (centered.T * weights) @ centered / total
    return center, np.linalg.eigh(covariance)[1][:, -1], used_soft_fallback


def axis_error(predicted: np.ndarray, truth_line: np.ndarray) -> float:
    truth = truth_line[1] - truth_line[0]
    cosine = abs(float(np.dot(predicted, truth))) / max(
        float(np.linalg.norm(predicted) * np.linalg.norm(truth)), 1e-12
    )
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def evaluate_lines(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    records: list[HVRecord],
    settings: dict,
    positive_weight: torch.Tensor,
    include_case_records: bool,
) -> tuple[dict, list[dict]]:
    model.eval()
    intersections = np.zeros(3, float)
    denominators = np.zeros(3, float)
    topology = np.zeros((3, 4), float)
    errors = [[], [], []]
    case_dice_by_class = [[], [], []]
    case_cldice_by_class = [[], [], []]
    case_macro_dice, case_macro_cldice = [], []
    soft_fallback_axes = 0
    case_rows = []
    total_loss, batches = 0.0, 0
    with torch.inference_mode():
        for features, masks, absolute_indices in loader:
            logits = model(features.to(device))
            loss, _ = line_loss(logits, masks.to(device), positive_weight, settings)
            total_loss += float(loss)
            batches += 1
            probabilities = torch.sigmoid(logits).cpu().numpy()
            truth = masks.numpy().astype(bool)
            prediction = probabilities >= float(settings["mask_threshold"])
            for batch_index, absolute in enumerate(absolute_indices.tolist()):
                record = records[int(absolute)]
                axis_directions = []
                per_case_errors = []
                per_case_dice = []
                per_case_cldice = []
                per_case_fallback = []
                for class_index in range(3):
                    predicted_mask = prediction[batch_index, class_index]
                    true_mask = truth[batch_index, class_index]
                    case_intersection = (
                        2.0 * np.logical_and(predicted_mask, true_mask).sum()
                    )
                    case_denominator = predicted_mask.sum() + true_mask.sum()
                    intersections[class_index] += case_intersection
                    denominators[class_index] += case_denominator
                    case_dice = float(case_intersection / max(case_denominator, 1e-12))
                    case_dice_by_class[class_index].append(case_dice)
                    per_case_dice.append(case_dice)
                    if not include_case_records:
                        continue
                    predicted_skeleton = morphological_skeleton(predicted_mask)
                    true_skeleton = morphological_skeleton(true_mask)
                    topology[class_index] += [
                        np.logical_and(predicted_skeleton, true_mask).sum(),
                        predicted_skeleton.sum(),
                        np.logical_and(true_skeleton, predicted_mask).sum(),
                        true_skeleton.sum(),
                    ]
                    topology_precision_case = np.logical_and(
                        predicted_skeleton, true_mask
                    ).sum() / max(predicted_skeleton.sum(), 1e-12)
                    topology_sensitivity_case = np.logical_and(
                        true_skeleton, predicted_mask
                    ).sum() / max(true_skeleton.sum(), 1e-12)
                    case_cldice = float(
                        2
                        * topology_precision_case
                        * topology_sensitivity_case
                        / max(
                            topology_precision_case + topology_sensitivity_case, 1e-12
                        )
                    )
                    case_cldice_by_class[class_index].append(case_cldice)
                    per_case_cldice.append(case_cldice)
                    resized_width, resized_height, left, top = letterbox_parameters(
                        record.width, record.height, int(math.sqrt(true_mask.size))
                    )
                    content = probabilities[
                        batch_index,
                        class_index,
                        top : top + resized_height,
                        left : left + resized_width,
                    ]
                    hard_content = predicted_mask[
                        top : top + resized_height, left : left + resized_width
                    ]
                    center, direction, used_soft_fallback = probability_axis(
                        content,
                        record.width,
                        record.height,
                        float(settings["axis_probability_power"]),
                        float(settings["axis_probability_quantile"]),
                        hard_content,
                        int(settings["minimum_hard_axis_pixels"]),
                    )
                    soft_fallback_axes += int(used_soft_fallback)
                    true_line = record.lines[class_index] * np.asarray(
                        [record.width, record.height], np.float32
                    )
                    error = axis_error(direction, true_line)
                    errors[class_index].append(error)
                    axis_directions.append(direction.tolist())
                    per_case_errors.append(error)
                    per_case_fallback.append(bool(used_soft_fallback))
                case_macro_dice.append(float(np.mean(per_case_dice)))
                if include_case_records:
                    case_macro_cldice.append(float(np.mean(per_case_cldice)))
                    case_rows.append(
                        {
                            "identifier": record.identifier,
                            "patient_id": record.patient_id,
                            **{
                                f"{name}_axis_dx": float(axis_directions[index][0])
                                for index, name in enumerate(LINE_NAMES)
                            },
                            **{
                                f"{name}_axis_dy": float(axis_directions[index][1])
                                for index, name in enumerate(LINE_NAMES)
                            },
                            **{
                                f"{name}_axis_error_deg": float(per_case_errors[index])
                                for index, name in enumerate(LINE_NAMES)
                            },
                            **{
                                f"{name}_dice": float(per_case_dice[index])
                                for index, name in enumerate(LINE_NAMES)
                            },
                            **{
                                f"{name}_cldice": float(per_case_cldice[index])
                                for index, name in enumerate(LINE_NAMES)
                            },
                            **{
                                f"{name}_soft_axis_fallback": bool(
                                    per_case_fallback[index]
                                )
                                for index, name in enumerate(LINE_NAMES)
                            },
                            "case_macro_dice": float(np.mean(per_case_dice)),
                            "case_macro_cldice": float(np.mean(per_case_cldice)),
                        }
                    )
    dice = intersections / np.maximum(denominators, 1e-12)
    if not include_case_records:
        metrics = {
            "loss": total_loss / max(batches, 1),
            "case_macro_dice": float(np.mean(case_macro_dice)),
            "pooled_class_macro_dice": float(dice.mean()),
        }
        for index, name in enumerate(LINE_NAMES):
            metrics[f"{name}_dice"] = float(dice[index])
        return metrics, []
    topology_precision = topology[:, 0] / np.maximum(topology[:, 1], 1e-12)
    topology_sensitivity = topology[:, 2] / np.maximum(topology[:, 3], 1e-12)
    cldice = (
        2
        * topology_precision
        * topology_sensitivity
        / np.maximum(topology_precision + topology_sensitivity, 1e-12)
    )
    class_axis = np.asarray([np.mean(values) for values in errors])
    metrics = {
        "loss": total_loss / max(batches, 1),
        "case_macro_dice": float(np.mean(case_macro_dice)),
        "case_macro_cldice": float(np.mean(case_macro_cldice)),
        "pooled_class_macro_dice": float(dice.mean()),
        "pooled_class_macro_cldice": float(cldice.mean()),
        "macro_axis_error_deg": float(class_axis.mean()),
        "soft_axis_fallback_count": int(soft_fallback_axes),
        "soft_axis_fallback_fraction": float(
            soft_fallback_axes / max(3 * len(case_rows), 1)
        ),
    }
    for index, name in enumerate(LINE_NAMES):
        metrics[f"{name}_dice"] = float(dice[index])
        metrics[f"{name}_cldice"] = float(cldice[index])
        metrics[f"{name}_case_mean_dice"] = float(np.mean(case_dice_by_class[index]))
        metrics[f"{name}_case_mean_cldice"] = float(
            np.mean(case_cldice_by_class[index])
        )
        metrics[f"{name}_axis_error_deg"] = float(class_axis[index])
    return metrics, case_rows


def train_line_model(
    model_id: str,
    spec: dict,
    splits: dict[str, list[HVRecord]],
    config: dict,
    output: Path,
    device: torch.device,
) -> dict:
    records = [row for split in SPLIT_NAMES for row in splits[split]]
    features, feature_manifest = dense_feature_cache(
        model_id, spec, records, config, output, device
    )
    masks, target_manifest = line_target_cache(records, config, output)
    indices, start = {}, 0
    for split in SPLIT_NAMES:
        indices[split] = np.arange(start, start + len(splits[split]), dtype=np.int64)
        start += len(splits[split])
    training_masks = np.asarray(masks[indices["train"]], np.uint8)
    positive = training_masks.sum(axis=(0, 2, 3)).astype(float)
    total = training_masks.shape[0] * training_masks.shape[2] * training_masks.shape[3]
    positive_weight_np = np.sqrt((total - positive) / np.maximum(positive, 1.0))
    positive_weight = torch.as_tensor(
        positive_weight_np, dtype=torch.float32, device=device
    )
    datasets = {
        name: LineFeatureDataset(features, masks, split_indices)
        for name, split_indices in indices.items()
    }
    settings = config["lines"]
    runs, prediction_rows = [], []
    checkpoint_dir = output / "heads" / "lines" / model_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in config["seeds"]:
        trials = []
        for trial_index, hyperparameters in enumerate(config["search_space"]):
            seed_everything(int(seed))
            generator = torch.Generator().manual_seed(int(seed))
            loaders = {
                name: DataLoader(
                    dataset,
                    batch_size=int(config["batch_size"]),
                    shuffle=name == "train",
                    num_workers=int(config["num_workers"]),
                    pin_memory=device.type == "cuda",
                    generator=generator if name == "train" else None,
                )
                for name, dataset in datasets.items()
            }
            model = LineDecoder(int(spec["dense_channels"]), settings).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=hyperparameters["lr"],
                weight_decay=hyperparameters["weight_decay"],
            )
            amp_enabled = bool(config.get("amp", False) and device.type == "cuda")
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            best_score, best_loss, best_epoch, best_state, stale = (
                -float("inf"),
                float("inf"),
                0,
                None,
                0,
            )
            epoch = 0
            for epoch in range(1, int(config["max_epochs"]) + 1):
                model.train()
                for feature_batch, mask_batch, _ in loaders["train"]:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(
                        device_type=device.type, enabled=amp_enabled
                    ):
                        logits = model(feature_batch.to(device))
                        loss, _ = line_loss(
                            logits, mask_batch.to(device), positive_weight, settings
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                validation, _ = evaluate_lines(
                    model,
                    loaders["val"],
                    device,
                    records,
                    settings,
                    positive_weight,
                    False,
                )
                score = validation["case_macro_dice"]
                validation_loss = validation["loss"]
                improved = score > best_score + 1e-8 or (
                    abs(score - best_score) <= 1e-8 and validation_loss < best_loss
                )
                if improved:
                    best_score, best_loss, best_epoch, stale = (
                        score,
                        validation_loss,
                        epoch,
                        0,
                    )
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                else:
                    stale += 1
                    if stale >= int(config["patience"]):
                        break
            if best_state is None:
                raise RuntimeError("line training produced no validation checkpoint")
            trials.append(
                {
                    "trial": trial_index,
                    "hyperparameters": hyperparameters,
                    "validation_case_macro_dice": best_score,
                    "validation_loss": best_loss,
                    "best_epoch": best_epoch,
                    "epochs_run": epoch,
                    "state": best_state,
                }
            )
        selected = max(
            trials,
            key=lambda row: (
                row["validation_case_macro_dice"],
                -row["validation_loss"],
            ),
        )
        model = LineDecoder(int(spec["dense_channels"]), settings).to(device)
        model.load_state_dict(selected["state"])
        checkpoint = checkpoint_dir / f"seed{seed}.pt"
        torch.save(selected["state"], checkpoint)
        test_loader = DataLoader(
            datasets["test"],
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        test_metrics, case_rows = evaluate_lines(
            model, test_loader, device, records, settings, positive_weight, True
        )
        for row in case_rows:
            prediction_rows.append({"model_id": model_id, "seed": seed, **row})
        clean_trials = [
            {key: value for key, value in trial.items() if key != "state"}
            for trial in trials
        ]
        runs.append(
            {
                "seed": int(seed),
                "selected_trial": int(selected["trial"]),
                "best_epoch": int(selected["best_epoch"]),
                "validation_case_macro_dice": float(
                    selected["validation_case_macro_dice"]
                ),
                "validation_loss": float(selected["validation_loss"]),
                "trials": clean_trials,
                "metrics": test_metrics,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    prediction_path = output / f"line_predictions_{model_id}.csv"
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
    fields = [
        "case_macro_dice",
        "case_macro_cldice",
        "pooled_class_macro_dice",
        "pooled_class_macro_cldice",
        "macro_axis_error_deg",
    ]
    for name in LINE_NAMES:
        fields.extend(
            [
                f"{name}_dice",
                f"{name}_cldice",
                f"{name}_case_mean_dice",
                f"{name}_case_mean_cldice",
                f"{name}_axis_error_deg",
            ]
        )
    return {
        "status": "complete",
        "display_name": spec.get("line_display_name", spec["display_name"]),
        "feature_manifest": feature_manifest,
        "target_manifest": target_manifest,
        "train_positive_weight": {
            name: float(value) for name, value in zip(LINE_NAMES, positive_weight_np)
        },
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in LineDecoder(
                int(spec["dense_channels"]), settings
            ).parameters()
        ),
        "runs": runs,
        "seed_summary": summarize_seed_metrics(runs, fields),
        "predictions": str(prediction_path.relative_to(ROOT)),
        "predictions_sha256": sha256_file(prediction_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--extract-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    chosen = args.models or list(config["models"])
    if not set(chosen) <= set(config["models"]):
        raise ValueError("Unknown model ID")
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits, audit = load_locked_records(config)
    records = [r for split in SPLIT_NAMES for r in splits[split]]
    masks, metadata = line_target_cache(records, config, output)
    positive = (
        np.asarray(masks[: len(splits["train"])], np.uint8)
        .sum(axis=(0, 2, 3))
        .astype(float)
    )
    total = len(splits["train"]) * config["image_size"] ** 2
    metadata["train_sqrt_imbalance_positive_weight"] = {
        name: float(math.sqrt((total - value) / max(value, 1)))
        for name, value in zip(LINE_NAMES, positive)
    }
    (output / "targets/line_masks.json").write_text(json.dumps(metadata, indent=2))
    if audit["image_dimension_mismatches"]:
        raise ValueError("Image dimensions differ from dataset annotations")
    print(
        json.dumps(
            {key: audit[key] for key in ["split_sizes", "split_patient_counts"]},
            indent=2,
        )
    )
    if args.audit_only:
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    result_path = output / "lines_results.json"
    identity = dict(
        config_sha256=sha256_json(config),
        implementation_sha256=sha256_file(Path(__file__)),
        dataset_csv_sha256=audit["dataset_csv_sha256"],
        angle_manifest_sha256=audit["angle_manifest_sha256"],
    )
    manifest = dict(
        protocol=config["protocol_version"],
        config=config,
        models={},
        baselines=line_template_baselines(splits, masks, config),
        **identity,
    )
    if result_path.exists():
        manifest = json.loads(result_path.read_text())
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError("Stale result manifest: use a new output directory")
    for model_id in chosen:
        spec = config["models"][model_id]
        pooled_feature_cache(model_id, spec, records, config, output, device)
        if args.extract_only:
            dense_feature_cache(model_id, spec, records, config, output, device)
            continue
        if manifest["models"].get(model_id, {}).get("status") != "complete":
            manifest["models"][model_id] = train_line_model(
                model_id, spec, splits, config, output, device
            )
            result_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

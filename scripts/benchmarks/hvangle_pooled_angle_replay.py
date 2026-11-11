#!/usr/bin/env python3
"""Joint HVA/IMA regression from frozen pooled encoder features (P-dir)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import hvangle_frozen_spatial_benchmark as spatial
import pooled_utils as endpoint


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("hvangle_pooled_angle_config.json")
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/benchmarks/controlled_cross_model_v2/medical/hvangle_pooled_angles_replay"
)
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


class AngleDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, indices: np.ndarray):
        self.features = features
        self.targets = targets
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        return (
            torch.from_numpy(self.features[index]),
            torch.from_numpy(self.targets[index]),
            index,
        )


class AngleHead(nn.Module):
    def __init__(self, pooled_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def icc_2_1(truth: np.ndarray, prediction: np.ndarray) -> float:
    values = np.column_stack([np.asarray(truth, float), np.asarray(prediction, float)])
    n, k = values.shape
    row_mean = values.mean(axis=1)
    column_mean = values.mean(axis=0)
    grand_mean = values.mean()
    ms_rows = k * np.square(row_mean - grand_mean).sum() / (n - 1)
    ms_columns = n * np.square(column_mean - grand_mean).sum() / (k - 1)
    residual = np.square(values - row_mean[:, None] - column_mean + grand_mean).sum()
    ms_error = residual / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else float("nan")


def measurement_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    difference = prediction - truth
    bias = float(difference.mean())
    sd = float(difference.std(ddof=1))
    return {
        "mae_deg": float(np.abs(difference).mean()),
        "icc_2_1": icc_2_1(truth, prediction),
        "bias_deg": bias,
        "loa_lower_deg": bias - 1.96 * sd,
        "loa_upper_deg": bias + 1.96 * sd,
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    predictions, indices = [], []
    for features, _, batch_indices in loader:
        predictions.append(model(features.to(device)).cpu().numpy())
        indices.append(np.asarray(batch_indices))
    return np.concatenate(predictions), np.concatenate(indices).astype(int)


def train_trial(
    features,
    targets,
    indices,
    config,
    model_id,
    seed,
    candidate_index,
    device,
    checkpoint_path=None,
):
    endpoint.seed_everything(seed)
    loaders = {
        split: DataLoader(
            AngleDataset(features, targets, indices[split]),
            batch_size=int(config["batch_size"]),
            shuffle=split == "train",
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for split in SPLITS
    }
    pooled_dim = int(config["models"][model_id]["pooled_dim"])
    hidden_dim = min(512, max(64, pooled_dim // 2))
    model = AngleHead(pooled_dim, hidden_dim).to(device)
    candidate = config["search_space"][candidate_index]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["lr"]),
        weight_decay=float(candidate["weight_decay"]),
    )
    criterion = nn.HuberLoss(delta=5.0, reduction="mean")
    best_state, best_validation, best_epoch, stale = None, math.inf, 0, 0
    start = time.time()
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        for batch_features, batch_targets, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features.to(device)), batch_targets.to(device))
            loss.backward()
            optimizer.step()
        validation_prediction, validation_order = predict(model, loaders["val"], device)
        score = float(
            np.abs(validation_prediction - targets[validation_order])
            .mean(axis=0)
            .mean()
        )
        if score < best_validation - 1e-10:
            best_validation, best_epoch, stale = score, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= int(config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("no angle checkpoint selected")
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": best_state,
                "model_id": model_id,
                "seed": seed,
                "pooled_dim": pooled_dim,
                "hidden_dim": hidden_dim,
                "validation_joint_mae_deg": best_validation,
            },
            checkpoint_path,
        )
    test_prediction, test_order = predict(model, loaders["test"], device)
    test_truth = targets[test_order]
    return (
        {
            "model_id": model_id,
            "encoder": config["models"][model_id]["display_name"],
            "seed": int(seed),
            "candidate_index": candidate_index,
            "candidate": candidate,
            "pooled_dim": pooled_dim,
            "hidden_dim": hidden_dim,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "best_epoch": best_epoch,
            "validation_joint_mae_deg": best_validation,
            "elapsed_seconds": time.time() - start,
            "test": {
                "hva": measurement_metrics(test_truth[:, 0], test_prediction[:, 0]),
                "ima": measurement_metrics(test_truth[:, 1], test_prediction[:, 1]),
            },
        },
        test_prediction,
        test_order,
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text())
    output = args.output.resolve()
    splits, split_audit = spatial.load_locked_records(config)
    records = endpoint.ordered_records(splits)
    indices = endpoint.split_indices(splits)
    targets = np.stack([record.targets for record in records]).astype(np.float32)
    chosen_models = args.models or list(config["models"])
    unknown = sorted(set(chosen_models) - set(config["models"]))
    if unknown:
        raise ValueError(f"unknown model IDs: {unknown}")
    feature_root = args.feature_root.resolve() if args.feature_root else None
    audit = {
        "protocol": "hvangle-frozen-pooled-angle-replay-v1",
        "config": str(args.config.resolve().relative_to(ROOT)),
        "config_sha256": endpoint.sha256_file(args.config.resolve()),
        "implementation": str(Path(__file__).resolve().relative_to(ROOT)),
        "implementation_sha256": endpoint.sha256_file(Path(__file__).resolve()),
        "selection_metric": "validation_joint_hva_ima_mae_deg",
        "loss": "mean Huber loss with delta=5 degrees",
        "split_sizes": {split: len(splits[split]) for split in SPLITS},
        "split_patient_counts": split_audit["split_patient_counts"],
        "seeds": config["seeds"],
        "search_space": config["search_space"],
        "feature_root_override": str(feature_root.relative_to(ROOT))
        if feature_root
        else None,
        "feature_sources": {},
    }
    feature_cache = {}
    for model_id in chosen_models:
        features, source = endpoint.load_features(
            config, model_id, len(records), feature_root
        )
        feature_cache[model_id] = features
        audit["feature_sources"][model_id] = source
    endpoint.atomic_json(output / "angle_replay_audit.json", audit)
    if not args.run:
        print(f"Audit passed; wrote {output / 'angle_replay_audit.json'}")
        return
    device = torch.device(args.device)
    trials, selected, prediction_rows = [], [], []
    test_records = splits["test"]
    for model_id, features in feature_cache.items():
        for seed in config["seeds"]:
            candidates = []
            candidate_predictions = []
            candidate_orders = []
            for candidate_index in range(len(config["search_space"])):
                result, prediction, order = train_trial(
                    features,
                    targets,
                    indices,
                    config,
                    model_id,
                    int(seed),
                    candidate_index,
                    device,
                    checkpoint_path=output
                    / "heads"
                    / model_id
                    / f"seed{seed}_trial{candidate_index}.pt",
                )
                trials.append(result)
                candidates.append(result)
                candidate_predictions.append(prediction)
                candidate_orders.append(order)
                print(
                    model_id,
                    seed,
                    candidate_index,
                    f"val={result['validation_joint_mae_deg']:.4f}",
                    f"HVA={result['test']['hva']['mae_deg']:.4f}",
                    f"IMA={result['test']['ima']['mae_deg']:.4f}",
                    flush=True,
                )
            winner = min(
                range(len(candidates)),
                key=lambda index: (
                    candidates[index]["validation_joint_mae_deg"],
                    index,
                ),
            )
            selected.append(candidates[winner])
            prediction = candidate_predictions[winner]
            order = candidate_orders[winner]
            for row_index, record in enumerate(test_records):
                prediction_rows.append(
                    {
                        "model_id": model_id,
                        "encoder": config["models"][model_id]["display_name"],
                        "seed": int(seed),
                        "identifier": record.identifier,
                        "patient_id": record.patient_id,
                        "reference_hva_deg": targets[order[row_index], 0],
                        "reference_ima_deg": targets[order[row_index], 1],
                        "predicted_hva_deg": prediction[row_index, 0],
                        "predicted_ima_deg": prediction[row_index, 1],
                    }
                )
    frame = pd.json_normalize(selected, sep=".")
    summary_rows = []
    metrics = (
        "test.hva.mae_deg",
        "test.hva.icc_2_1",
        "test.hva.bias_deg",
        "test.hva.loa_lower_deg",
        "test.hva.loa_upper_deg",
        "test.ima.mae_deg",
        "test.ima.icc_2_1",
        "test.ima.bias_deg",
        "test.ima.loa_lower_deg",
        "test.ima.loa_upper_deg",
    )
    for model_id, group in frame.groupby("model_id", sort=False):
        row = {
            "model_id": model_id,
            "encoder": group["encoder"].iloc[0],
            "trainable_parameters": int(group["trainable_parameters"].iloc[0]),
        }
        for metric in metrics:
            short = metric.removeprefix("test.").replace(".", "_")
            row[f"{short}_mean"] = float(group[metric].mean())
            row[f"{short}_sd"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    endpoint.atomic_json(
        output / "angle_replay_results.json",
        {
            "protocol": audit["protocol"],
            "created_unix": time.time(),
            "audit_sha256": endpoint.sha256_file(output / "angle_replay_audit.json"),
            "trial_results": trials,
            "selected_results": selected,
        },
    )
    endpoint.atomic_csv(output / "angle_replay_summary.csv", pd.DataFrame(summary_rows))
    endpoint.atomic_csv(
        output / "angle_replay_predictions.csv", pd.DataFrame(prediction_rows)
    )
    print(f"Wrote angle replay results to {output}")


if __name__ == "__main__":
    main()

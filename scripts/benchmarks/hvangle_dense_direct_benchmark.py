#!/usr/bin/env python3
"""Direct angle readout from the exact archived dense geometry features.

No encoder execution, geometry supervision, PCA, or angle protocol is used.
Only train/validation rows participate in candidate fitting/selection; test
inference runs once per selected seed. Sources are opened read-only.
"""

from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import hvangle_frozen_spatial_benchmark as spatial

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("hvangle_dense_direct_config.json")
PROTOCOL = "hvangle-frozen-dense-direct-v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


class DenseAngleHead(nn.Module):
    """Same projection and four upsampling blocks as the dense mask head.

    Replace the final 16->3 mask projection with spatial 14x14 pooling and a
    linear two-angle output. The head retains spatial position and adds 6,223
    parameters relative to LineDecoder; no anatomical labels are consumed.
    """

    def __init__(self, channels, settings):
        super().__init__()
        geometry = spatial.LineDecoder(channels, settings)
        self.grid = geometry.grid
        self.project = geometry.project
        self.blocks = geometry.blocks
        width = int(settings["decoder_channels"][-1])
        self.readout = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.grid, self.grid)),
            nn.Flatten(),
            nn.Linear(width * self.grid**2, 2),
        )

    def forward(self, features):
        x = nn.functional.interpolate(
            features, size=(self.grid, self.grid), mode="bilinear", align_corners=False
        )
        x = self.project(x)
        for block in self.blocks:
            x = nn.functional.interpolate(
                x, scale_factor=2.0, mode="bilinear", align_corners=False
            )
            x = block(x)
        return self.readout(x)


class DenseDataset(Dataset):
    def __init__(self, features, targets, indices):
        self.features, self.targets, self.indices = features, targets, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        # A copy keeps torch writes away from the read-only memmap.
        return (
            torch.from_numpy(np.array(self.features[i], dtype=np.float32)),
            torch.from_numpy(self.targets[i].copy()),
            i,
        )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def audit(config, models):
    source = (ROOT / config["source_root"]).resolve()
    manifest_path = source / config["angle_manifest"]
    csv_path = source / config["dataset_csv"]
    locked = json.loads(manifest_path.read_text())
    split_ids = locked["split_ids"]
    order = [x for key in ["train", "val", "test"] for x in split_ids[key]]
    assert len(order) == len(set(order)) == 1177
    frame = pd.read_csv(csv_path)
    frame = frame[frame.filename.isin(order)].copy()
    assert len(frame) == 1177 and frame.filename.is_unique
    table = frame.set_index("filename").loc[order].copy()
    targets = table[["HVA", "IMA"]].to_numpy(dtype=np.float32)
    assert targets.shape == (1177, 2) and np.isfinite(targets).all()
    indices = {}
    offset = 0
    patients = {}
    for key, count in [("train", 829), ("val", 172), ("test", 176)]:
        assert len(split_ids[key]) == count
        indices[key] = np.arange(offset, offset + count)
        offset += count
        patients[key] = set(table.iloc[indices[key]].patient_id.astype(int))
    assert [len(patients[k]) for k in indices] == [617, 132, 133]
    assert all(
        not patients[a] & patients[b]
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    )
    csv_sha, angle_sha = digest(csv_path), digest(manifest_path)
    ids_sha = spatial.sha256_ids(order)
    caches, sources = {}, {}
    for model in models:
        folder = source / config["models"][model]["cache"]
        feature_file, meta_file = folder / "dense.npy", folder / "dense.json"
        metadata = json.loads(meta_file.read_text())
        assert metadata["ids_sha256"] == ids_sha, model
        assert metadata["dataset_csv_sha256"] == csv_sha, model
        assert metadata["angle_manifest_sha256"] == angle_sha, model
        assert (
            metadata["model_id"] == model
            and metadata["per_location_l2_normalized"] is True
        )
        features = np.load(feature_file, mmap_mode="r")
        assert list(features.shape) == metadata["shape"] and features.shape[0] == 1177
        assert str(features.dtype) == metadata["dtype"] == "float16"
        actual_sha = digest(feature_file)
        assert actual_sha == metadata["feature_sha256"], model
        assert np.isfinite(features[[0, 828, 829, 1000, 1001, 1176]]).all()
        caches[model] = features
        head = DenseAngleHead(features.shape[1], config["head"])
        original = spatial.LineDecoder(features.shape[1], config["head"])
        params, geom_params = (
            sum(p.numel() for p in head.parameters()),
            sum(p.numel() for p in original.parameters()),
        )
        assert params - geom_params == 6223
        sources[model] = {
            "feature_file": str(feature_file),
            "feature_sha256": actual_sha,
            "feature_manifest_sha256": digest(meta_file),
            "shape": list(features.shape),
            "encoder_checkpoint_sha256": metadata["encoder_source"]["sha256"],
            "interface": metadata["interface"],
            "parameters": params,
            "geometry_parameters": geom_params,
        }
    evidence = {
        "protocol": PROTOCOL,
        "status": "passed",
        "models": models,
        "split_sizes": {k: len(v) for k, v in indices.items()},
        "split_patient_counts": {k: len(v) for k, v in patients.items()},
        "ordered_ids_sha256": ids_sha,
        "dataset_csv_sha256": csv_sha,
        "angle_manifest_sha256": angle_sha,
        "feature_sources": sources,
        "head": "dense geometry trunk + AdaptiveAvgPool(14x14) + Flatten + Linear(3136,2)",
        "loss": "joint Huber(delta=5 degrees)",
        "selection": "minimum validation joint HVA/IMA MAE",
        "geometry_labels_used": False,
        "test_used_for_selection": False,
        "seeds": config["seeds"],
        "search_space": config["search_space"],
        "max_epochs": config["max_epochs"],
        "patience": config["patience"],
        "amp": config["amp"],
        "gradient_clip_norm": config["gradient_clip_norm"],
        "shared_decoder_implementation_sha256": digest(spatial.__file__),
        "implementation_sha256": digest(__file__),
        "config_sha256": digest(config["_config_file"]),
    }
    return caches, targets, indices, table, evidence


def loader(features, targets, index, config, train=False):
    return DataLoader(
        DenseDataset(features, targets, index),
        batch_size=config["batch_size"],
        shuffle=train,
        num_workers=0,
        pin_memory=False,
    )


@torch.inference_mode()
def predict(model, data, device, amp):
    model.eval()
    output, rows = [], []
    for x, _, i in data:
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            y = model(x.to(device))
        output.append(y.float().cpu().numpy())
        rows.append(i.numpy())
    return np.concatenate(output), np.concatenate(rows)


def train_trial(
    features, targets, indices, config, model_id, seed, trial, output, device, identity
):
    path = output / "trials" / model_id / f"seed{seed}_candidate{trial}.pt"
    if path.exists():
        stored = torch.load(path, map_location="cpu", weights_only=False)
        assert stored["identity"] == identity, f"Incompatible resume checkpoint: {path}"
        return stored
    seed_all(seed)
    head = DenseAngleHead(features.shape[1], config["head"]).to(device)
    candidate = config["search_space"][trial]
    optim = torch.optim.AdamW(
        head.parameters(), lr=candidate["lr"], weight_decay=candidate["weight_decay"]
    )
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.HuberLoss(delta=5.0)
    train = loader(features, targets, indices["train"], config, True)
    val = loader(features, targets, indices["val"], config)
    best, best_state, best_epoch, stale = math.inf, None, 0, 0
    start = time.time()
    for epoch in range(1, config["max_epochs"] + 1):
        head.train()
        losses = []
        for x, y, _ in train:
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                pred = head(x.to(device))
                loss = criterion(pred, y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(head.parameters(), config["gradient_clip_norm"])
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.detach()))
        prediction, order = predict(head, val, device, amp)
        score = float(np.abs(prediction - targets[order]).mean())
        if not math.isfinite(score):
            raise FloatingPointError("non-finite validation metric")
        if score < best - 1e-10:
            best, best_epoch, stale = score, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }
        else:
            stale += 1
        progress = {
            "status": "training",
            "model_id": model_id,
            "seed": seed,
            "candidate": trial,
            "epoch": epoch,
            "train_huber": float(np.mean(losses)),
            "validation_joint_mae": score,
            "best_validation_joint_mae": best,
            "elapsed_seconds": time.time() - start,
            "updated_unix": time.time(),
        }
        atomic_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        if stale >= config["patience"]:
            break
    stored = {
        "identity": identity,
        "state": best_state,
        "model_id": model_id,
        "seed": seed,
        "candidate_index": trial,
        "hyperparameters": candidate,
        "best_epoch": best_epoch,
        "validation_joint_mae_deg": best,
        "epochs_run": epoch,
        "elapsed_seconds": time.time() - start,
    }
    atomic_save(path, stored)
    return stored


def write_results(output, results, rows):
    clean = [{k: v for k, v in r.items() if k != "state"} for r in results]
    atomic_json(
        output / "dense_direct_results.json",
        {"protocol": PROTOCOL, "selected_results": clean},
    )
    for filename, data in [("dense_direct_predictions.csv", rows)]:
        if data:
            temp = output / (filename + ".tmp")
            pd.DataFrame(data).to_csv(temp, index=False)
            os.replace(temp, output / filename)
    summary = []
    for model in dict.fromkeys(r["model_id"] for r in clean):
        selected = [r for r in clean if r["model_id"] == model]
        record = {"model_id": model, "completed_seeds": len(selected)}
        for key in ["hva_mae_deg", "ima_mae_deg"]:
            values = [r[key] for r in selected]
            record[key + "_mean"] = float(np.mean(values))
            record[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        summary.append(record)
    temp = output / "dense_direct_summary.csv.tmp"
    pd.DataFrame(summary).to_csv(temp, index=False)
    os.replace(temp, output / "dense_direct_summary.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--models", nargs="+")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--smoke-test-heads", action="store_true")
    mode.add_argument("--run", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    config["_config_file"] = str(args.config.resolve())
    models = args.models or list(config["models"])
    assert set(models) <= set(config["models"])
    output = (args.output or ROOT / config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    if args.run:
        lock = (output / "training.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    caches, targets, indices, table, evidence = audit(config, models)
    atomic_json(output / "dense_direct_audit.json", evidence)
    print(
        json.dumps(
            {
                "audit": "passed",
                "models": models,
                "split_sizes": evidence["split_sizes"],
            }
        ),
        flush=True,
    )
    if args.audit_only:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    device = torch.device(args.device)
    amp = config["amp"] and device.type == "cuda"
    torch.set_num_threads(4)
    if args.smoke_test_heads:
        for model, features in caches.items():
            seed_all(17)
            head = DenseAngleHead(features.shape[1], config["head"]).to(device)
            x, y, _ = next(
                iter(loader(features, targets, indices["train"][:2], config))
            )
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                pred = head(x.to(device))
                loss = nn.HuberLoss(delta=5.0)(pred, y.to(device))
            assert pred.shape == (2, 2) and torch.isfinite(loss)
            loss.backward()
            assert all(
                torch.isfinite(q.grad).all()
                for q in head.parameters()
                if q.grad is not None
            )
            print(
                json.dumps(
                    {
                        "smoke": "passed",
                        "model": model,
                        "shape": list(pred.shape),
                        "loss": float(loss.detach()),
                        "parameters": sum(q.numel() for q in head.parameters()),
                    }
                ),
                flush=True,
            )
            del head
            torch.cuda.empty_cache()
        return
    identity = {
        k: evidence[k]
        for k in [
            "config_sha256",
            "implementation_sha256",
            "shared_decoder_implementation_sha256",
            "ordered_ids_sha256",
            "dataset_csv_sha256",
            "angle_manifest_sha256",
        ]
    }
    identity["feature_sha256"] = {
        k: v["feature_sha256"] for k, v in evidence["feature_sources"].items()
    }
    runfile = output / "run_identity.json"
    if runfile.exists():
        assert (
            json.loads(runfile.read_text()) == identity
        ), "Run identity changed; use a fresh output directory"
    else:
        atomic_json(runfile, identity)
    results = []
    rows = []
    try:
        for model, features in caches.items():
            for seed in config["seeds"]:
                completed = output / "selected" / model / f"seed{seed}.json"
                predictions = (
                    output / "selected" / model / f"seed{seed}_predictions.csv"
                )
                if completed.exists() and predictions.exists():
                    result = json.loads(completed.read_text())
                    assert result["identity"] == identity
                    results.append(result)
                    rows.extend(pd.read_csv(predictions).to_dict("records"))
                    continue
                trials = [
                    train_trial(
                        features,
                        targets,
                        indices,
                        config,
                        model,
                        seed,
                        k,
                        output,
                        device,
                        identity,
                    )
                    for k in range(len(config["search_space"]))
                ]
                winner = min(
                    trials,
                    key=lambda r: (r["validation_joint_mae_deg"], r["candidate_index"]),
                )
                head = DenseAngleHead(features.shape[1], config["head"]).to(device)
                head.load_state_dict(winner["state"])
                pred, order = predict(
                    head,
                    loader(features, targets, indices["test"], config),
                    device,
                    amp,
                )
                assert np.array_equal(order, indices["test"])
                errors = np.abs(pred - targets[order]).mean(0)
                result = {k: v for k, v in winner.items() if k != "state"}
                result.update(
                    hva_mae_deg=float(errors[0]),
                    ima_mae_deg=float(errors[1]),
                    parameters=sum(q.numel() for q in head.parameters()),
                )
                predrows = [
                    {
                        "model_id": model,
                        "seed": seed,
                        "identifier": table.index[i],
                        "patient_id": int(table.iloc[i].patient_id),
                        "reference_hva_deg": float(targets[i, 0]),
                        "reference_ima_deg": float(targets[i, 1]),
                        "predicted_hva_deg": float(v[0]),
                        "predicted_ima_deg": float(v[1]),
                    }
                    for i, v in zip(order, pred)
                ]
                completed.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(predrows).to_csv(predictions, index=False)
                atomic_json(completed, result)
                results.append(result)
                rows.extend(predrows)
                write_results(output, results, rows)
                print(
                    json.dumps(
                        {
                            "selected": model,
                            "seed": seed,
                            "candidate": winner["candidate_index"],
                            "hva_mae_deg": result["hva_mae_deg"],
                            "ima_mae_deg": result["ima_mae_deg"],
                        }
                    ),
                    flush=True,
                )
                del head
                torch.cuda.empty_cache()
        atomic_json(
            output / "progress.json",
            {
                "status": "complete",
                "selected_runs": len(results),
                "expected_runs": len(models) * len(config["seeds"]),
                "updated_unix": time.time(),
            },
        )
    except BaseException as exc:
        atomic_json(
            output / "failure.json", {"error": repr(exc), "updated_unix": time.time()}
        )
        raise


if __name__ == "__main__":
    main()

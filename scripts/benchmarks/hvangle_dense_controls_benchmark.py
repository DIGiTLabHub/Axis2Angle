#!/usr/bin/env python3
"""Two validation-selected controls on archived, frozen HVAngleEst features.

aux_mask: original dense-direct head plus the geometric route's mask task.
grid_direct: identical dense-direct parameters, but all convolutions stay at
14x14 (no 224x224 upsampling followed by pooling back to 14x14).

The auxiliary weight is fixed at 1 before fitting, not optimized on test.
The same three LR/WD candidates and seeds as the original control are used.
Sources are read-only. Neither variant executes PCA to produce its answer.
"""

from __future__ import annotations
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import hvangle_dense_direct_benchmark as direct

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("hvangle_dense_controls_config.json")
PROTOCOL = "hvangle-frozen-dense-review-controls-v1"


class AuxiliaryMaskHead(direct.DenseAngleHead):
    def __init__(self, channels, settings):
        super().__init__(channels, settings)
        # Add after the original head so its initialization stays identical.
        # Preserve the original CPU RNG stream (and thus training batch order).
        with torch.random.fork_rng(devices=[]):
            self.mask_output = nn.Conv2d(int(settings["decoder_channels"][-1]), 3, 1)

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
        return self.readout(x), self.mask_output(x)


class GridDirectHead(direct.DenseAngleHead):
    def forward(self, features):
        x = nn.functional.interpolate(
            features, size=(self.grid, self.grid), mode="bilinear", align_corners=False
        )
        x = self.project(x)
        for block in self.blocks:
            x = block(x)
        # Pool is the identity at the 14x14 grid; same angle-head parameters.
        return self.readout(x), None


def make_head(variant, channels, settings):
    return {"aux_mask": AuxiliaryMaskHead, "grid_direct": GridDirectHead}[variant](
        channels, settings
    )


class ControlDataset(Dataset):
    def __init__(self, features, targets, indices, masks=None):
        self.features, self.targets, self.indices, self.masks = (
            features,
            targets,
            indices,
            masks,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        mask = (
            torch.from_numpy(np.array(self.masks[i], dtype=np.float32))
            if self.masks is not None
            else torch.empty(0)
        )
        return (
            torch.from_numpy(np.array(self.features[i], dtype=np.float32)),
            torch.from_numpy(self.targets[i].copy()),
            mask,
            i,
        )


def loader(features, targets, indices, config, masks=None, train=False):
    return DataLoader(
        ControlDataset(features, targets, indices, masks),
        batch_size=config["batch_size"],
        shuffle=train,
        num_workers=0,
        pin_memory=False,
    )


def audit(config, models, variants):
    caches, targets, indices, table, evidence = direct.audit(config, models)
    folder = (ROOT / config["source_root"]).resolve() / config["mask_cache"]
    path, meta_path = folder / "line_masks.npy", folder / "line_masks.json"
    meta = json.loads(meta_path.read_text())
    assert meta["ids_sha256"] == evidence["ordered_ids_sha256"]
    assert meta["dataset_csv_sha256"] == evidence["dataset_csv_sha256"]
    assert meta["line_width_px"] == 3.0
    assert direct.digest(path) == meta["target_sha256"]
    masks = np.load(path, mmap_mode="r")
    assert masks.shape == (1177, 3, 224, 224) and masks.dtype == np.uint8
    # Verify target-coordinate convention against the archived renderer.
    for i in [0, 414, 828, 829, 1000, 1001, 1176]:
        row = table.iloc[i]
        record = SimpleNamespace(
            width=int(row.image_width),
            height=int(row.image_height),
            lines=np.stack(
                [
                    direct.spatial.parse_numbers(row[k], 4).reshape(2, 2)
                    for k in direct.spatial.LINE_NAMES
                ]
            ),
        )
        assert np.array_equal(
            direct.spatial.render_line_masks(record, 224, 3.0), masks[i]
        )
    # Class weights use only the training partition, never all-image counts.
    positive = (
        np.asarray(masks[indices["train"]], dtype=np.uint8)
        .sum(axis=(0, 2, 3))
        .astype(float)
    )
    total = len(indices["train"]) * 224 * 224
    weights = np.sqrt((total - positive) / np.maximum(positive, 1.0)).astype(np.float32)
    expected = np.array(
        [
            meta["train_sqrt_imbalance_positive_weight"][k]
            for k in direct.spatial.LINE_NAMES
        ]
    )
    assert np.allclose(weights, expected, rtol=1e-6)
    evidence.update(
        protocol=PROTOCOL,
        variants=variants,
        auxiliary_weight=config["auxiliary_mask_weight"],
        auxiliary_weight_selected_on_test=False,
        mask_targets_sha256=meta["target_sha256"],
        mask_manifest_sha256=direct.digest(meta_path),
        mask_targets=str(path),
        train_positive_weights=weights.tolist(),
        mask_loss_accumulation="float32 sigmoid and reductions, including under AMP",
        original_dense_direct_implementation_sha256=direct.digest(direct.__file__),
        implementation_sha256=direct.digest(__file__),
        variant_definitions={
            "aux_mask": "same dense-direct trunk and angle head plus 16->3 mask projection; Huber + fixed-weight original BCE/Dice",
            "grid_direct": "same dense-direct parameters; project and four convolutions all at canonical 14x14, no upsampling",
        },
        geometry_labels_used={"aux_mask": True, "grid_direct": False},
        head="variant-specific; see variant_definitions",
        loss="Huber(delta=5); aux_mask additionally uses fixed-weight .5 weighted BCE + .5 soft Dice",
        selection="minimum validation joint HVA/IMA MAE, for both variants",
        expected_trials=len(models)
        * len(variants)
        * len(config["seeds"])
        * len(config["search_space"]),
        expected_selected_runs=len(models) * len(variants) * len(config["seeds"]),
    )
    for model, features in caches.items():
        params = {}
        for variant in variants:
            h = make_head(variant, features.shape[1], config["head"])
            params[variant] = sum(p.numel() for p in h.parameters())
        evidence["feature_sources"][model]["control_parameters"] = params
        original = evidence["feature_sources"][model]["parameters"]
        if "grid_direct" in params:
            assert params["grid_direct"] == original
        if "aux_mask" in params:
            assert params["aux_mask"] == original + 51
    return caches, targets, indices, table, masks, weights, evidence


@torch.inference_mode()
def predict(head, data, device, amp):
    head.eval()
    predictions = []
    rows = []
    for x, _, _, i in data:
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            prediction, _ = head(x.to(device))
        predictions.append(prediction.float().cpu().numpy())
        rows.append(i.numpy())
    return np.concatenate(predictions), np.concatenate(rows)


def auxiliary_loss(logits, truth, weights, settings):
    # Accumulate sigmoid/Dice in float32: a full 224x224 batch exceeds
    # float16's finite summation range. The mathematical loss is unchanged.
    return direct.spatial.line_loss(logits.float(), truth.float(), weights, settings)


def smoke(caches, targets, indices, masks, weights, config, variants, device):
    results = []
    probe = torch.zeros(
        (16, 3, 224, 224), dtype=torch.float16, device=device, requires_grad=True
    )
    truth = torch.from_numpy(
        np.array(masks[indices["train"][:16]], dtype=np.float32)
    ).to(device)
    probe_loss, parts = auxiliary_loss(
        probe, truth, torch.as_tensor(weights, device=device), config["mask_loss"]
    )
    assert torch.isfinite(probe_loss) and 0 < parts["soft_dice_loss"] < 1
    probe_loss.backward()
    assert torch.isfinite(probe.grad).all()
    del probe, truth, probe_loss
    for model, features in caches.items():
        x, y, mask, _ = next(
            iter(loader(features, targets, indices["train"][:2], config, masks))
        )
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        for variant in variants:
            direct.seed_all(17)
            base = direct.DenseAngleHead(features.shape[1], config["head"]).to(device)
            original_rng = torch.get_rng_state().clone()
            direct.seed_all(17)
            head = make_head(variant, features.shape[1], config["head"]).to(device)
            assert torch.equal(original_rng, torch.get_rng_state())
            assert all(
                torch.equal(v, head.state_dict()[k])
                for k, v in base.state_dict().items()
            )
            angle, logits = head(x)
            assert angle.shape == (2, 2)
            if variant == "aux_mask":
                assert logits.shape == (2, 3, 224, 224)
                assert torch.allclose(angle, base(x), atol=1e-6, rtol=1e-6)
                seg, _ = auxiliary_loss(
                    logits,
                    mask,
                    torch.as_tensor(weights, device=device),
                    config["mask_loss"],
                )
                seg.backward(retain_graph=True)
                assert head.project[0].weight.grad.abs().sum() > 0
                assert head.mask_output.weight.grad.abs().sum() > 0
                assert head.readout[-1].weight.grad is None
                head.zero_grad(set_to_none=True)
            loss = nn.HuberLoss(delta=5.0)(angle, y)
            if logits is not None:
                loss = loss + config["auxiliary_mask_weight"] * seg
            assert torch.isfinite(loss)
            loss.backward()
            assert head.readout[-1].weight.grad.abs().sum() > 0
            assert all(
                torch.isfinite(p.grad).all()
                for p in head.parameters()
                if p.grad is not None
            )
            results.append(
                dict(
                    model=model,
                    variant=variant,
                    status="passed",
                    finite_loss=float(loss.detach()),
                    original_parameters=sum(p.numel() for p in base.parameters()),
                    parameters=sum(p.numel() for p in head.parameters()),
                    shared_initialization_matches=True,
                    original_rng_stream_preserved=True,
                )
            )
            del head, base, angle, logits, loss
        del x, y, mask
        torch.cuda.empty_cache()
    return results


def train_trial(
    features,
    targets,
    indices,
    masks,
    weights,
    config,
    model_id,
    variant,
    seed,
    trial,
    output,
    device,
    identity,
):
    path = output / "trials" / variant / model_id / f"seed{seed}_candidate{trial}.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert (
            state["identity"] == identity
        ), "Incompatible completed trial; use fresh output"
        return state
    direct.seed_all(seed)
    head = make_head(variant, features.shape[1], config["head"]).to(device)
    hp = config["search_space"][trial]
    optim = torch.optim.AdamW(
        head.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    amp = bool(config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    weight = torch.as_tensor(weights, device=device)
    criterion = nn.HuberLoss(delta=5.0)
    train = loader(
        features,
        targets,
        indices["train"],
        config,
        masks if variant == "aux_mask" else None,
        True,
    )
    # Validation loader never reads mask labels; model selection is angle-only.
    val = loader(features, targets, indices["val"], config)
    best, best_state, best_epoch, stale = math.inf, None, 0, 0
    start = time.time()
    for epoch in range(1, config["max_epochs"] + 1):
        head.train()
        angle_losses = []
        mask_losses = []
        for x, y, mask, _ in train:
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                pred, logits = head(x.to(device))
                angle_loss = criterion(pred, y.to(device))
                mask_loss = angle_loss.new_zeros(())
                if variant == "aux_mask":
                    mask_loss, _ = auxiliary_loss(
                        logits, mask.to(device), weight, config["mask_loss"]
                    )
                loss = angle_loss + config["auxiliary_mask_weight"] * mask_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(head.parameters(), config["gradient_clip_norm"])
            scaler.step(optim)
            scaler.update()
            angle_losses.append(float(angle_loss.detach()))
            mask_losses.append(float(mask_loss.detach()))
        pred, order = predict(head, val, device, amp)
        score = float(np.abs(pred - targets[order]).mean())
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite validation MAE")
        if score < best - 1e-10:
            best, best_epoch, stale = score, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }
        else:
            stale += 1
        progress = dict(
            status="training",
            variant=variant,
            model_id=model_id,
            seed=seed,
            candidate=trial,
            epoch=epoch,
            train_angle_huber=float(np.mean(angle_losses)),
            train_mask_loss=float(np.mean(mask_losses)),
            validation_joint_mae=score,
            best_validation_joint_mae=best,
            elapsed_seconds=time.time() - start,
            updated_unix=time.time(),
        )
        direct.atomic_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        if stale >= config["patience"]:
            break
    result = dict(
        identity=identity,
        state=best_state,
        variant=variant,
        model_id=model_id,
        seed=seed,
        candidate_index=trial,
        hyperparameters=hp,
        best_epoch=best_epoch,
        validation_joint_mae_deg=best,
        epochs_run=epoch,
        elapsed_seconds=time.time() - start,
    )
    direct.atomic_save(path, result)
    return result


def write_results(output, results, rows):
    direct.atomic_json(
        output / "controls_results.json",
        {"protocol": PROTOCOL, "selected_results": results},
    )
    temp = output / "controls_predictions.csv.tmp"
    pd.DataFrame(rows).to_csv(temp, index=False)
    os.replace(temp, output / "controls_predictions.csv")
    records = []
    for (variant, model), group in pd.DataFrame(results).groupby(
        ["variant", "model_id"], sort=False
    ):
        record = dict(variant=variant, model_id=model, completed_seeds=len(group))
        for metric in ["hva_mae_deg", "ima_mae_deg"]:
            record[metric + "_mean"] = float(group[metric].mean())
            record[metric + "_sd"] = (
                float(group[metric].std()) if len(group) > 1 else None
            )
        records.append(record)
    temp = output / "controls_summary.csv.tmp"
    pd.DataFrame(records).to_csv(temp, index=False)
    os.replace(temp, output / "controls_summary.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--models", nargs="+")
    p.add_argument("--variants", nargs="+", choices=["aux_mask", "grid_direct"])
    mode = p.add_mutually_exclusive_group(required=True)
    for name in ["audit-only", "smoke-test-heads", "run"]:
        mode.add_argument("--" + name, action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    config["_config_file"] = str(args.config.resolve())
    models = args.models or list(config["models"])
    variants = args.variants or config["variants"]
    assert set(models) <= set(config["models"])
    assert config["protocol"] == PROTOCOL
    output = (args.output or ROOT / config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    if args.run:
        lock = (output / "training.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    caches, targets, indices, table, masks, weights, evidence = audit(
        config, models, variants
    )
    direct.atomic_json(output / "controls_audit.json", evidence)
    print(
        json.dumps(
            dict(
                audit="passed",
                models=models,
                variants=variants,
                expected_trials=evidence["expected_trials"],
            )
        ),
        flush=True,
    )
    if args.audit_only:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    if args.smoke_test_heads:
        result = smoke(
            caches, targets, indices, masks, weights, config, variants, device
        )
        direct.atomic_json(
            output / "smoke_validation.json", {"status": "passed", "checks": result}
        )
        print(json.dumps(result), flush=True)
        return
    identity = {
        k: evidence[k]
        for k in [
            "config_sha256",
            "implementation_sha256",
            "shared_decoder_implementation_sha256",
            "original_dense_direct_implementation_sha256",
            "ordered_ids_sha256",
            "dataset_csv_sha256",
            "angle_manifest_sha256",
            "mask_targets_sha256",
        ]
    }
    identity["feature_sha256"] = {
        k: v["feature_sha256"] for k, v in evidence["feature_sources"].items()
    }
    identity["variants"] = variants
    runfile = output / "run_identity.json"
    if runfile.exists():
        assert (
            json.loads(runfile.read_text()) == identity
        ), "Run identity changed; use fresh output"
    else:
        direct.atomic_json(runfile, identity)
    results = []
    rows = []
    try:
        # Finish the primary supervision control before the architecture check.
        for variant in variants:
            for model, features in caches.items():
                for seed in config["seeds"]:
                    folder = output / "selected" / variant / model
                    done = folder / f"seed{seed}.json"
                    predfile = folder / f"seed{seed}_predictions.csv"
                    if done.exists() and predfile.exists():
                        r = json.loads(done.read_text())
                        assert r["identity"] == identity
                        results.append(r)
                        rows.extend(pd.read_csv(predfile).to_dict("records"))
                        continue
                    trials = [
                        train_trial(
                            features,
                            targets,
                            indices,
                            masks,
                            weights,
                            config,
                            model,
                            variant,
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
                        key=lambda r: (
                            r["validation_joint_mae_deg"],
                            r["candidate_index"],
                        ),
                    )
                    head = make_head(variant, features.shape[1], config["head"]).to(
                        device
                    )
                    head.load_state_dict(winner["state"])
                    # First test inference for this selected seed follows all candidate selection.
                    pred, order = predict(
                        head,
                        loader(features, targets, indices["test"], config),
                        device,
                        bool(config["amp"] and device.type == "cuda"),
                    )
                    assert np.array_equal(order, indices["test"])
                    error = np.abs(pred - targets[order]).mean(0)
                    r = {k: v for k, v in winner.items() if k != "state"}
                    r.update(
                        hva_mae_deg=float(error[0]),
                        ima_mae_deg=float(error[1]),
                        parameters=sum(p.numel() for p in head.parameters()),
                    )
                    predrows = [
                        dict(
                            variant=variant,
                            model_id=model,
                            seed=seed,
                            identifier=table.index[i],
                            patient_id=int(table.iloc[i].patient_id),
                            reference_hva_deg=float(targets[i, 0]),
                            reference_ima_deg=float(targets[i, 1]),
                            predicted_hva_deg=float(v[0]),
                            predicted_ima_deg=float(v[1]),
                        )
                        for i, v in zip(order, pred)
                    ]
                    folder.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(predrows).to_csv(predfile, index=False)
                    direct.atomic_json(done, r)
                    results.append(r)
                    rows.extend(predrows)
                    write_results(output, results, rows)
                    print(
                        json.dumps(
                            dict(
                                selected=variant,
                                model=model,
                                seed=seed,
                                hva_mae_deg=r["hva_mae_deg"],
                                ima_mae_deg=r["ima_mae_deg"],
                            )
                        ),
                        flush=True,
                    )
                    del head
                    torch.cuda.empty_cache()
        direct.atomic_json(
            output / "progress.json",
            dict(
                status="complete",
                selected_runs=len(results),
                expected_runs=evidence["expected_selected_runs"],
                updated_unix=time.time(),
            ),
        )
    except BaseException as exc:
        direct.atomic_json(
            output / "failure.json", dict(error=repr(exc), updated_unix=time.time())
        )
        raise


if __name__ == "__main__":
    main()

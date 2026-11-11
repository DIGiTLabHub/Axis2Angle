#!/usr/bin/env python3
"""Frozen-dense Gaussian endpoint heatmaps -> soft-argmax -> fixed angle protocol.

The mask decoder trunk is unchanged; its output becomes six logits maps.
Training uses endpoint-order-invariant Gaussian distribution cross-entropy.
Angle labels are used for validation selection, not for training loss.
"""

from __future__ import annotations
import argparse, fcntl, json, math, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import hvangle_dense_direct_benchmark as base

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("hvangle_dense_heatmap_config.json")
PROTOCOL = "hvangle-frozen-dense-endpoint-heatmap-v1"
AXES = ["ph1", "m1", "m2"]


class HeatmapHead(base.spatial.LineDecoder):
    def __init__(self, channels, settings):
        super().__init__(channels, settings)
        # Preserve the inherited trunk's initialization and CPU RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.output = nn.Conv2d(int(settings["decoder_channels"][-1]), 6, 1)


def spatial_log_prob(logits, temperature):
    return torch.log_softmax(logits.float().flatten(2) / temperature, dim=-1)


def soft_argmax(logits, temperature=1.0):
    b, c, h, w = logits.shape
    p = spatial_log_prob(logits, temperature).exp().reshape(b, c, h, w)
    x = torch.arange(w, device=logits.device, dtype=torch.float32) + 0.5
    y = torch.arange(h, device=logits.device, dtype=torch.float32) + 0.5
    return torch.stack(
        [(p.sum(-2) * x).sum(-1), (p.sum(-1) * y).sum(-1)], dim=-1
    ).reshape(b, 3, 2, 2)


def gaussian_targets(points, size, sigma):
    points = points.float().reshape(-1, 6, 2)
    grid = torch.arange(size, device=points.device, dtype=torch.float32) + 0.5
    px = torch.softmax(
        -0.5 * ((grid[None, None] - points[:, :, 0, None]) / sigma) ** 2, dim=-1
    )
    py = torch.softmax(
        -0.5 * ((grid[None, None] - points[:, :, 1, None]) / sigma) ** 2, dim=-1
    )
    return (py[:, :, :, None] * px[:, :, None, :]).flatten(2)


def endpoint_loss(logits, points, settings):
    with torch.amp.autocast(device_type=logits.device.type, enabled=False):
        logp = spatial_log_prob(logits, settings["temperature"]).reshape(
            -1, 3, 2, logits.shape[-1] * logits.shape[-2]
        )
        q = gaussian_targets(points, logits.shape[-1], settings["sigma_px"]).reshape_as(
            logp
        )
        # Cost[case,axis,predicted endpoint,reference endpoint]. Match independently
        # per anatomical axis; swapping endpoints changes neither geometry nor loss.
        cost = -torch.einsum("bapn,batn->bapt", logp, q)
        forward = cost[:, :, 0, 0] + cost[:, :, 1, 1]
        reverse = cost[:, :, 0, 1] + cost[:, :, 1, 0]
        return torch.minimum(forward, reverse).mean() / 2


def acute_angle(a, b):
    return np.degrees(
        np.arctan2(
            np.abs(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]),
            np.abs((a * b).sum(-1)),
        )
    )


def directions(points, epsilon):
    v = points[:, :, 1] - points[:, :, 0]
    length = np.linalg.norm(v, axis=-1)
    bad = length < epsilon
    unit = v / np.maximum(length[..., None], epsilon)
    unit[bad] = [1.0, 0.0]
    return unit, bad


def decode_numpy(canvas, offset, scale, epsilon):
    original = (canvas - offset[:, None, None, :]) * scale[:, None, None, :]
    unit, bad = directions(original, epsilon)
    angles = np.stack(
        [acute_angle(unit[:, 0], unit[:, 1]), acute_angle(unit[:, 1], unit[:, 2])],
        axis=-1,
    )
    return original, unit, angles, bad


class EndpointData(Dataset):
    def __init__(self, features, points, indices):
        self.features, self.points, self.indices = features, points, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        return (
            torch.from_numpy(np.array(self.features[i], dtype=np.float32)),
            torch.from_numpy(self.points[i].astype(np.float32)),
            i,
        )


def loader(features, points, indices, config, train=False, seed=17):
    return DataLoader(
        EndpointData(features, points, indices),
        batch_size=config["batch_size"],
        shuffle=train,
        num_workers=0,
        pin_memory=False,
        generator=torch.Generator().manual_seed(seed) if train else None,
    )


def audit(config, models):
    caches, _, indices, table, evidence = base.audit(config, models)
    original = np.stack(
        [
            np.stack(
                [
                    np.fromstring(row[name], sep=",", dtype=np.float64).reshape(2, 2)
                    for name in base.spatial.LINE_NAMES
                ]
            )
            for _, row in table.iterrows()
        ]
    )
    assert (
        original.shape == (1177, 3, 2, 2)
        and np.isfinite(original).all()
        and (original >= 0).all()
        and (original <= 1).all()
    )
    canvas = []
    offset = []
    scale = []
    source = []
    for i, (_, row) in enumerate(table.iterrows()):
        w, h = int(row.image_width), int(row.image_height)
        rw, rh, left, top = base.spatial.letterbox_parameters(
            w, h, config["heatmap"]["size"]
        )
        canvas.append(original[i] * [rw, rh] + [left, top])
        offset.append([left, top])
        scale.append([w / rw, h / rh])
        source.append(original[i] * [w, h])
    canvas = np.stack(canvas)
    offset = np.asarray(offset, dtype=float)
    scale = np.asarray(scale)
    source = np.stack(source)
    gt_axes, bad = directions(
        source, config["heatmap"]["collapsed_axis_epsilon_source_px"]
    )
    assert not bad.any()
    targets = table[["HVA", "IMA"]].to_numpy(dtype=np.float64)
    reconstructed = np.stack(
        [
            acute_angle(gt_axes[:, 0], gt_axes[:, 1]),
            acute_angle(gt_axes[:, 1], gt_axes[:, 2]),
        ],
        -1,
    )
    mismatch = float(np.abs(reconstructed - targets).max())
    assert mismatch < 1e-8
    assert np.allclose(
        (canvas - offset[:, None, None]) * scale[:, None, None], source, atol=1e-10
    )
    evidence.update(
        protocol=PROTOCOL,
        head="original dense mask trunk with six endpoint logits maps at 224x224",
        loss="mean per-axis minimum of two endpoint assignments, normalized Gaussian spatial cross-entropy",
        selection="minimum validation joint HVA/IMA MAE",
        geometry_labels_used=True,
        angle_labels_used_in_training_loss=False,
        reference_angle_reconstruction_max_deg=mismatch,
        heatmap=config["heatmap"],
        coordinate_convention="pixel centers x+0.5/y+0.5; expectation in float32; inverse letterbox before geometry",
        minimum_canvas_coordinate=float(canvas.min()),
        maximum_canvas_coordinate=float(canvas.max()),
        minimum_reference_axis_source_px=float(
            np.linalg.norm(source[:, :, 1] - source[:, :, 0], axis=-1).min()
        ),
        collapsed_axis_rule="predicted source length < epsilon: horizontal unit vector, same for scoring and execution; never drop test cases",
        parameterization_fixed_before_first_heatmap_test=True,
        architecture_search_performed=False,
        implementation_sha256=base.digest(__file__),
        base_implementation_sha256=base.digest(base.__file__),
        expected_candidates=len(models)
        * len(config["seeds"])
        * len(config["search_space"]),
        expected_selected_runs=len(models) * len(config["seeds"]),
    )
    for model, features in caches.items():
        head = HeatmapHead(features.shape[1], config["head"])
        n = sum(p.numel() for p in head.parameters())
        assert n == evidence["feature_sources"][model]["geometry_parameters"] + 51
        evidence["feature_sources"][model]["heatmap_parameters"] = n
    return (
        caches,
        canvas,
        offset,
        scale,
        source,
        gt_axes,
        targets,
        indices,
        table,
        evidence,
    )


@torch.inference_mode()
def predict(head, data, device, config):
    head.eval()
    coords = []
    rows = []
    for x, _, index in data:
        with torch.amp.autocast(
            device_type=device.type, enabled=config["amp"] and device.type == "cuda"
        ):
            logits = head(x.to(device))
        points = soft_argmax(logits, config["heatmap"]["temperature"])
        coords.append(points.cpu().numpy().astype(float))
        rows.append(index.numpy())
    return np.concatenate(coords), np.concatenate(rows)


def evaluate_coords(canvas, index, offset, scale, source, gt_axes, targets, config):
    pts, axes, angles, collapsed = decode_numpy(
        canvas,
        offset[index],
        scale[index],
        config["heatmap"]["collapsed_axis_epsilon_source_px"],
    )
    axis_error = acute_angle(axes, gt_axes[index])
    # Unordered endpoint distance in canonical canvas pixels, not source pixels.
    truth = (source[index] / scale[index, None, None]) + offset[index, None, None]
    cost_a = np.linalg.norm(canvas - truth, axis=-1).mean(-1)
    cost_b = np.linalg.norm(canvas - truth[:, :, ::-1], axis=-1).mean(-1)
    endpoint_error = np.minimum(cost_a, cost_b)
    assert np.isfinite(angles).all() and np.isfinite(axis_error).all()
    return dict(
        points_source=pts,
        axes=axes,
        angles=angles,
        collapsed=collapsed,
        axis_error=axis_error,
        endpoint_error=endpoint_error,
        angle_error=np.abs(angles - targets[index]),
    )


def smoke(
    caches, canvas, offset, scale, source, gt_axes, targets, indices, config, device
):
    # Subpixel recovery, inverse letterboxing, endpoint symmetry, and gradients.
    sample = np.array(
        [
            [
                [[30.25, 40.75], [50.6, 80.4]],
                [[80.2, 90.3], [95.7, 130.6]],
                [[140.4, 100.1], [145.8, 170.2]],
            ]
        ],
        dtype=np.float32,
    )
    q = gaussian_targets(
        torch.tensor(sample, device=device), 224, config["heatmap"]["sigma_px"]
    )
    logits = q.clamp_min(1e-38).log().reshape(1, 6, 224, 224)
    recovered = soft_argmax(logits, 1.0).cpu().numpy()
    assert np.max(np.abs(recovered - sample)) < 1e-4
    points = torch.tensor(sample, device=device)
    l1 = endpoint_loss(logits, points, config["heatmap"])
    l2 = endpoint_loss(logits, points.flip(2), config["heatmap"])
    assert torch.allclose(l1, l2, atol=1e-6)
    trial = np.random.default_rng(17).normal(size=(2, 3, 2, 2))
    trial[:, :, 1] += 2
    ax, _ = directions(trial, 1e-8)
    ax2, _ = directions(trial[:, :, ::-1], 1e-8)
    assert np.allclose(
        acute_angle(ax[:, 0], ax[:, 1]), acute_angle(ax2[:, 0], ax2[:, 1])
    )
    z = np.ones((2, 3, 2, 2))
    unit, bad = directions(z, 1e-8)
    assert bad.all() and np.array_equal(unit, np.broadcast_to([1.0, 0.0], unit.shape))
    exact = evaluate_coords(
        canvas[indices["val"]],
        indices["val"],
        offset,
        scale,
        source,
        gt_axes,
        targets,
        config,
    )
    assert exact["angle_error"].max() < 1e-8
    checks = []
    for model, features in caches.items():
        base.seed_all(17)
        original = base.spatial.LineDecoder(features.shape[1], config["head"])
        rng = torch.get_rng_state().clone()
        base.seed_all(17)
        head = HeatmapHead(features.shape[1], config["head"]).to(device)
        assert torch.equal(rng, torch.get_rng_state())
        assert all(
            torch.equal(v, head.state_dict()[k].cpu())
            for k, v in original.state_dict().items()
            if not k.startswith("output.")
        )
        x, y, _ = next(iter(loader(features, canvas, indices["train"][:2], config)))
        with torch.amp.autocast(
            device_type=device.type, enabled=config["amp"] and device.type == "cuda"
        ):
            out = head(x.to(device))
            loss = endpoint_loss(out, y.to(device), config["heatmap"])
        assert out.shape == (2, 6, 224, 224) and torch.isfinite(loss)
        loss.backward()
        assert head.project[0].weight.grad.abs().sum() > 0
        assert all(
            torch.isfinite(p.grad).all()
            for p in head.parameters()
            if p.grad is not None
        )
        coords = soft_argmax(out.detach(), config["heatmap"]["temperature"])
        assert torch.isfinite(coords).all()
        checks.append(
            dict(
                model=model,
                status="passed",
                parameters=sum(p.numel() for p in head.parameters()),
                finite_loss=float(loss.detach()),
                shape=list(out.shape),
            )
        )
        del head, original, out, loss, x, y
        torch.cuda.empty_cache()
    return dict(
        status="passed",
        subpixel_recovery_max_px=float(np.abs(recovered - sample).max()),
        reference_angle_reconstruction=True,
        swap_invariance=True,
        collapsed_axis_rule=True,
        unchanged_trunk_initialization_and_rng=True,
        models=checks,
    )


def train_trial(
    features,
    canvas,
    offset,
    scale,
    source,
    gt_axes,
    targets,
    indices,
    config,
    model,
    seed,
    candidate,
    output,
    device,
    identity,
):
    path = output / "trials" / model / f"seed{seed}_candidate{candidate}.pt"
    if path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["identity"] == identity
        return checkpoint
    base.seed_all(seed)
    head = HeatmapHead(features.shape[1], config["head"]).to(device)
    hp = config["search_space"][candidate]
    optim = torch.optim.AdamW(
        head.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    amp = bool(config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train = loader(features, canvas, indices["train"], config, True, seed)
    val = loader(features, canvas, indices["val"], config)
    best, best_state, best_epoch, stale = math.inf, None, 0, 0
    start = time.time()
    for epoch in range(1, config["max_epochs"] + 1):
        head.train()
        losses = []
        for x, y, _ in train:
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                pred = head(x.to(device))
                loss = endpoint_loss(pred, y.to(device), config["heatmap"])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite heatmap loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(head.parameters(), config["gradient_clip_norm"])
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.detach()))
        predicted, index = predict(head, val, device, config)
        v = evaluate_coords(
            predicted, index, offset, scale, source, gt_axes, targets, config
        )
        score = float(v["angle_error"].mean())
        if not math.isfinite(score):
            raise FloatingPointError("non-finite validation angle MAE")
        if score < best - 1e-10:
            best, best_epoch, stale = score, epoch, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }
        else:
            stale += 1
        progress = dict(
            status="training",
            model_id=model,
            seed=seed,
            candidate=candidate,
            epoch=epoch,
            train_heatmap_ce=float(np.mean(losses)),
            validation_joint_mae=score,
            best_validation_joint_mae=best,
            validation_axis_mae=float(v["axis_error"].mean()),
            validation_collapsed_axes=int(v["collapsed"].sum()),
            elapsed_seconds=time.time() - start,
            updated_unix=time.time(),
        )
        base.atomic_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        if stale >= config["patience"]:
            break
    checkpoint = dict(
        identity=identity,
        state=best_state,
        model_id=model,
        seed=seed,
        candidate_index=candidate,
        hyperparameters=hp,
        best_epoch=best_epoch,
        validation_joint_mae_deg=best,
        epochs_run=epoch,
        elapsed_seconds=time.time() - start,
    )
    base.atomic_save(path, checkpoint)
    return checkpoint


def write_results(output, results, rows):
    base.atomic_json(
        output / "heatmap_results.json",
        dict(protocol=PROTOCOL, selected_results=results),
    )
    temp = output / "heatmap_predictions.csv.tmp"
    pd.DataFrame(rows).to_csv(temp, index=False)
    os.replace(temp, output / "heatmap_predictions.csv")
    frame = pd.DataFrame(results)
    summary = []
    for model, g in frame.groupby("model_id", sort=False):
        row = dict(model_id=model, completed_seeds=len(g))
        for k in [
            "hva_mae_deg",
            "ima_mae_deg",
            "axis_mae_deg",
            "endpoint_error_canvas_px",
        ]:
            row[k + "_mean"] = float(g[k].mean())
            row[k + "_sd"] = float(g[k].std()) if len(g) > 1 else None
        summary.append(row)
    temp = output / "heatmap_summary.csv.tmp"
    pd.DataFrame(summary).to_csv(temp, index=False)
    os.replace(temp, output / "heatmap_summary.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--models", nargs="+")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    for name in ["audit-only", "smoke-test-heads", "run"]:
        mode.add_argument("--" + name, action="store_true")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    config["_config_file"] = str(args.config.resolve())
    models = args.models or list(config["models"])
    assert set(models) <= set(config["models"]) and config["protocol"] == PROTOCOL
    output = (args.output or ROOT / config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    if args.run:
        lock = (output / "training.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = audit(config, models)
    (
        caches,
        canvas,
        offset,
        scale,
        source,
        gt_axes,
        targets,
        indices,
        table,
        evidence,
    ) = data
    base.atomic_json(output / "heatmap_audit.json", evidence)
    print(
        json.dumps(
            dict(
                audit="passed",
                models=models,
                expected_candidates=evidence["expected_candidates"],
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
            caches,
            canvas,
            offset,
            scale,
            source,
            gt_axes,
            targets,
            indices,
            config,
            device,
        )
        base.atomic_json(output / "smoke_validation.json", result)
        print(json.dumps(result), flush=True)
        return
    identity = {
        k: evidence[k]
        for k in [
            "config_sha256",
            "implementation_sha256",
            "base_implementation_sha256",
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
        ), "Identity changed; use a new output"
    else:
        base.atomic_json(runfile, identity)
    results = []
    rows = []
    try:
        for model, features in caches.items():
            for seed in config["seeds"]:
                folder = output / "selected" / model
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
                        canvas,
                        offset,
                        scale,
                        source,
                        gt_axes,
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
                head = HeatmapHead(features.shape[1], config["head"]).to(device)
                head.load_state_dict(winner["state"])
                # Test inference only after all three candidates and validation selection.
                pred, index = predict(
                    head,
                    loader(features, canvas, indices["test"], config),
                    device,
                    config,
                )
                assert np.array_equal(index, indices["test"])
                v = evaluate_coords(
                    pred, index, offset, scale, source, gt_axes, targets, config
                )
                r = {k: value for k, value in winner.items() if k != "state"}
                r.update(
                    hva_mae_deg=float(v["angle_error"][:, 0].mean()),
                    ima_mae_deg=float(v["angle_error"][:, 1].mean()),
                    axis_mae_deg=float(v["axis_error"].mean()),
                    endpoint_error_canvas_px=float(v["endpoint_error"].mean()),
                    collapsed_axes=int(v["collapsed"].sum()),
                    parameters=sum(p.numel() for p in head.parameters()),
                )
                predrows = []
                for b, i in enumerate(index):
                    row = dict(
                        model_id=model,
                        seed=seed,
                        identifier=str(table.index[i]),
                        patient_id=int(table.iloc[i].patient_id),
                        reference_hva_deg=float(targets[i, 0]),
                        reference_ima_deg=float(targets[i, 1]),
                        predicted_hva_deg=float(v["angles"][b, 0]),
                        predicted_ima_deg=float(v["angles"][b, 1]),
                        axis_mae_deg=float(v["axis_error"][b].mean()),
                        endpoint_error_canvas_px=float(v["endpoint_error"][b].mean()),
                    )
                    for a, name in enumerate(AXES):
                        row.update(
                            {
                                f"{name}_axis_dx": float(v["axes"][b, a, 0]),
                                f"{name}_axis_dy": float(v["axes"][b, a, 1]),
                                f"{name}_axis_error_deg": float(v["axis_error"][b, a]),
                                f"{name}_collapsed": bool(v["collapsed"][b, a]),
                            }
                        )
                        for ep in range(2):
                            for xy, dim in enumerate(["x", "y"]):
                                row[f"{name}_e{ep}_{dim}_canvas"] = float(
                                    pred[b, a, ep, xy]
                                )
                                row[f"{name}_e{ep}_{dim}_source"] = float(
                                    v["points_source"][b, a, ep, xy]
                                )
                    predrows.append(row)
                folder.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(predrows).to_csv(predfile, index=False)
                base.atomic_json(done, r)
                results.append(r)
                rows.extend(predrows)
                write_results(output, results, rows)
                print(
                    json.dumps(
                        dict(
                            selected=model,
                            seed=seed,
                            hva_mae_deg=r["hva_mae_deg"],
                            ima_mae_deg=r["ima_mae_deg"],
                            axis_mae_deg=r["axis_mae_deg"],
                        )
                    ),
                    flush=True,
                )
                del head
                torch.cuda.empty_cache()
        base.atomic_json(
            output / "progress.json",
            dict(
                status="complete",
                selected_runs=len(results),
                expected_runs=len(models) * len(config["seeds"]),
                updated_unix=time.time(),
            ),
        )
    except BaseException as e:
        base.atomic_json(
            output / "failure.json", dict(error=repr(e), updated_unix=time.time())
        )
        raise


if __name__ == "__main__":
    main()

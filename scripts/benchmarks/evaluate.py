#!/usr/bin/env python3
"""Evaluate axis recovery, measurement utility, and reference-axis substitution."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hvangle_frozen_spatial_benchmark as spatial

SEEDS = [17, 42, 73]
ANGLES = ["HVA", "IMA"]


def angle(a, b):
    return np.degrees(
        np.arctan2(
            np.abs(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]),
            np.abs((a * b).sum(-1)),
        )
    )


def measure(axes):
    return np.stack(
        [
            angle(axes[..., 0, :], axes[..., 1, :]),
            angle(axes[..., 1, :], axes[..., 2, :]),
        ],
        axis=-1,
    )


def bootstrap(differences, patients, seed=20260826):
    """Image-weighted patient-cluster intervals; average seeds before resampling."""
    unique, inverse = np.unique(
        np.asarray(patients).astype(np.int64), return_inverse=True
    )
    counts = np.bincount(inverse)
    draws = np.random.default_rng(seed).integers(len(unique), size=(5000, len(unique)))
    sums = np.stack(
        [
            np.bincount(inverse, weights=differences[:, j], minlength=len(unique))
            for j in range(differences.shape[1])
        ],
        axis=1,
    )
    samples = sums[draws].sum(1) / counts[draws].sum(1)[:, None]
    return np.quantile(
        samples, [0.025, 0.975, 0.05 / (2 * 12), 1 - 0.05 / (2 * 12)], axis=0
    )


def load_seed_frames(frame, model, ids):
    selected = frame[frame.model_id.eq(model)]
    if set(selected.seed.unique()) != set(SEEDS):
        raise ValueError(f"{model}: expected exactly seeds {SEEDS}")
    result = []
    for seed in SEEDS:
        part = selected[selected.seed.eq(seed)].set_index("identifier")
        if not part.index.is_unique or set(part.index) != set(ids):
            raise ValueError(
                f"{model}/{seed}: prediction identifiers differ from test split"
            )
        result.append(part.loc[ids])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-roots",
        type=Path,
        nargs="+",
        required=True,
        help="M-mask output directories containing line_predictions_MODEL.csv",
    )
    parser.add_argument(
        "--direct",
        action="append",
        default=[],
        metavar="ROUTE=CSV",
        help="Repeat for each complete six-encoder direct prediction table",
    )
    parser.add_argument("--dataset", type=Path, default=spatial.DATASET_CSV)
    parser.add_argument(
        "--split",
        type=Path,
        default=spatial.ROOT
        / "outputs/benchmarks/controlled_cross_model_v2/medical/hvangle/hvangle_results.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids = json.loads(args.split.read_text())["split_ids"]["test"]
    metadata = pd.read_csv(args.dataset).set_index("filename").loc[ids]
    patients = metadata.patient_id.to_numpy(dtype=np.int64)
    reference = metadata[ANGLES].to_numpy(dtype=float)
    reference_axes = np.stack(
        [
            np.stack(
                [
                    spatial.parse_numbers(row[k], 4).reshape(2, 2)
                    for k in spatial.LINE_NAMES
                ]
            )
            for _, row in metadata.iterrows()
        ]
    )
    reference_axes *= metadata[["image_width", "image_height"]].to_numpy()[
        :, None, None, :
    ]
    reference_axes = reference_axes[:, :, 1] - reference_axes[:, :, 0]
    if not np.allclose(measure(reference_axes), reference, atol=1e-3):
        raise ValueError("Reference axes do not reproduce dataset angles")
    direct_tables = {}
    for item in args.direct:
        route, path = item.split("=", 1)
        if route in direct_tables:
            raise ValueError(
                f"Duplicate route: {route}; concatenate per-run tables first"
            )
        table = pd.read_csv(path)
        if "variant" in table:
            variant = {"G-dir": "grid_direct", "A-dir": "aux_mask"}.get(route, route)
            table = table[table.variant.eq(variant)]
            if table.empty:
                raise ValueError(f"No rows for variant {variant}")
        direct_tables[route] = table
    summaries = []
    contrasts = []
    substitutions = []
    recovery = []
    models = set()
    for folder in args.mask_roots:
        for path in sorted(folder.glob("line_predictions_*.csv")):
            frame = pd.read_csv(path)
            model = frame.model_id.iloc[0]
            if model in models:
                raise ValueError(f"Duplicate mask predictions: {model}")
            models.add(model)
            parts = load_seed_frames(frame, model, ids)
            for part in parts:
                if not np.array_equal(part.patient_id.to_numpy(), patients):
                    raise ValueError(
                        "Mask prediction patient IDs differ from annotations"
                    )
            axes = np.stack(
                [
                    np.stack(
                        [
                            p[[f"{k}_axis_dx", f"{k}_axis_dy"]].to_numpy()
                            for k in spatial.LINE_NAMES
                        ],
                        axis=1,
                    )
                    for p in parts
                ]
            )
            predictions = {"M-mask": measure(axes)}
            values = np.array(
                [
                    p[[f"{k}_axis_error_deg" for k in spatial.LINE_NAMES]]
                    .to_numpy()
                    .mean()
                    for p in parts
                ]
            )
            recovery.append(
                dict(
                    model_id=model,
                    axis="macro",
                    mae=values.mean(),
                    seed_sd=values.std(ddof=1),
                )
            )
            for k in spatial.LINE_NAMES:
                values = np.array([p[f"{k}_axis_error_deg"].mean() for p in parts])
                recovery.append(
                    dict(
                        model_id=model,
                        axis=k,
                        mae=values.mean(),
                        seed_sd=values.std(ddof=1),
                    )
                )
            for route, table in direct_tables.items():
                direct_parts = load_seed_frames(table, model, ids)
                for part in direct_parts:
                    if not np.array_equal(part.patient_id.to_numpy(), patients):
                        raise ValueError(
                            "Direct prediction patient IDs differ from annotations"
                        )
                    if not np.allclose(
                        part[["reference_hva_deg", "reference_ima_deg"]],
                        reference,
                        atol=1e-5,
                    ):
                        raise ValueError("Direct references differ from annotations")
                predictions[route] = np.stack(
                    [
                        p[["predicted_hva_deg", "predicted_ima_deg"]].to_numpy()
                        for p in direct_parts
                    ]
                )
            errors = {k: np.abs(v - reference[None]) for k, v in predictions.items()}
            for route, error in errors.items():
                values = error.mean(axis=1)
                for j, target in enumerate(ANGLES):
                    summaries.append(
                        dict(
                            model_id=model,
                            route=route,
                            angle=target,
                            mae=values[:, j].mean(),
                            seed_sd=values[:, j].std(ddof=1),
                        )
                    )
            for route in direct_tables:
                delta = (errors["M-mask"] - errors[route]).mean(axis=0)
                bounds = bootstrap(delta, patients)
                for j, target in enumerate(ANGLES):
                    contrasts.append(
                        dict(
                            model_id=model,
                            comparator=route,
                            angle=target,
                            difference=delta[:, j].mean(),
                            ci95_low=bounds[0, j],
                            ci95_high=bounds[1, j],
                            adjusted_low=bounds[2, j],
                            adjusted_high=bounds[3, j],
                        )
                    )
            for index, k in enumerate(spatial.LINE_NAMES):
                replaced = axes.copy()
                replaced[:, :, index] = reference_axes[None, :, index]
                changed = np.abs(measure(replaced) - reference[None])
                gain = (errors["M-mask"] - changed).mean(axis=0)
                bounds = bootstrap(gain, patients)
                for j, target in enumerate(ANGLES):
                    substitutions.append(
                        dict(
                            model_id=model,
                            axis=k,
                            angle=target,
                            substituted_mae=changed[:, :, j].mean(),
                            gain=gain[:, j].mean(),
                            exploratory_ci95_low=bounds[0, j],
                            exploratory_ci95_high=bounds[1, j],
                        )
                    )
    if not models:
        raise ValueError("No mask predictions found")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("recovery", recovery),
        ("measurement", summaries),
        ("contrasts", contrasts),
        ("substitution", substitutions),
    ]:
        if rows:
            pd.DataFrame(rows).to_csv(args.output / f"{name}.csv", index=False)
    print(
        f"Evaluated {len(models)} encoders, {len(ids)} test images, {len(set(patients))} patients."
    )


if __name__ == "__main__":
    main()

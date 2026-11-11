#!/usr/bin/env python3
"""Recreate the patient-disjoint HVAngleEst split used in the paper."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/benchmarks/controlled_cross_model_v2/medical/hvangle/hvangle_results.json"
)


def make_split(data_root):
    frame = pd.read_csv(data_root / "HVAngleEst/datasets.csv")
    frame = frame[~frame.filename.duplicated(keep=False)]
    names = set().union(
        *(
            set(pd.read_csv(data_root / "splits" / f"{s}.csv").filename)
            for s in ("train", "val", "test")
        )
    )
    frame = frame[frame.filename.isin(names)]
    patients = np.asarray(sorted(frame.patient_id.unique()))
    patients = patients[np.random.RandomState(42).permutation(len(patients))]
    n_train, n_val = int(0.70 * len(patients)), int(0.15 * len(patients))
    groups = dict(
        train=patients[:n_train],
        val=patients[n_train : n_train + n_val],
        test=patients[n_train + n_val :],
    )
    split = {
        s: frame[frame.patient_id.isin(p)].filename.tolist() for s, p in groups.items()
    }
    if [len(split[s]) for s in groups] != [829, 172, 176]:
        raise ValueError("Dataset does not reproduce the 829/172/176 image split")
    if [len(p) for p in groups.values()] != [617, 132, 133]:
        raise ValueError("Dataset does not reproduce the 617/132/133 patient split")
    return {"split_ids": split}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "HVAngleEst")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = make_split(args.data_root)
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        if existing["split_ids"] != result["split_ids"]:
            raise ValueError("Existing split differs; refusing to overwrite")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Verified: 829/172/176 images; 617/132/133 disjoint patients.")


if __name__ == "__main__":
    main()

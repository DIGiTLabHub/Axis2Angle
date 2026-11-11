import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/benchmarks"))
import hvangle_dense_controls_benchmark as controls
import hvangle_dense_direct_benchmark as direct
import hvangle_dense_heatmap_benchmark as heatmap
import hvangle_frozen_spatial_benchmark as spatial
import hvangle_pooled_angle_replay as pooled


@pytest.fixture
def small_problem():
    torch.set_num_threads(1)
    rng = np.random.default_rng(17)
    features = rng.normal(size=(8, 8, 2, 2)).astype(np.float32)
    targets = np.column_stack([np.arange(8) + 20, np.arange(8) + 10]).astype(np.float32)
    indices = dict(train=np.arange(4), val=np.arange(4, 6), test=np.arange(6, 8))
    config = dict(
        head=dict(
            canonical_feature_grid=1, projection_channels=8, decoder_channels=[8] * 4
        ),
        batch_size=2,
        max_epochs=1,
        patience=1,
        amp=False,
        gradient_clip_norm=1.0,
        search_space=[dict(lr=1e-4, weight_decay=1e-4)],
        auxiliary_mask_weight=1.0,
        mask_loss=dict(bce_loss_weight=0.5, soft_dice_loss_weight=0.5),
        heatmap=dict(
            size=16,
            temperature=1.0,
            sigma_px=2.0,
            collapsed_axis_epsilon_source_px=1e-8,
        ),
    )
    return features, targets, indices, config


@pytest.mark.parametrize(
    "route", ["D-dir", "G-dir", "A-dir", "H-end", "P-dir", "M-mask"]
)
def test_training_checkpoint_and_prediction(
    route, small_problem, tmp_path, monkeypatch
):
    features, targets, indices, config = small_problem
    device = torch.device("cpu")
    masks = np.zeros((8, 3, 16, 16), dtype=np.uint8)
    masks[:, :, 3:13, 7:9] = 1
    if route == "D-dir":
        result = direct.train_trial(
            features, targets, indices, config, "test", 17, 0, tmp_path, device, {}
        )
        head = direct.DenseAngleHead(8, config["head"])
        head.load_state_dict(result["state"])
        predictions, order = direct.predict(
            head,
            direct.loader(features, targets, indices["test"], config),
            device,
            False,
        )
    elif route in ["G-dir", "A-dir"]:
        variant = {"G-dir": "grid_direct", "A-dir": "aux_mask"}[route]
        result = controls.train_trial(
            features,
            targets,
            indices,
            masks,
            np.ones(3, dtype=np.float32),
            config,
            "test",
            variant,
            17,
            0,
            tmp_path,
            device,
            {},
        )
        head = controls.make_head(variant, 8, config["head"])
        head.load_state_dict(result["state"])
        predictions, order = controls.predict(
            head,
            controls.loader(features, targets, indices["test"], config),
            device,
            False,
        )
    elif route == "P-dir":
        config["models"] = {"test": dict(pooled_dim=8, display_name="Test")}
        result, predictions, order = pooled.train_trial(
            features.mean((2, 3)),
            targets,
            indices,
            config,
            "test",
            17,
            0,
            device,
            checkpoint_path=tmp_path / "pooled.pt",
        )
        assert torch.load(tmp_path / "pooled.pt", weights_only=False)["state"]
    elif route == "H-end":
        rng = np.random.default_rng(42)
        points = rng.uniform(2, 14, size=(8, 3, 2, 2))
        offset = np.zeros((8, 2))
        scale = np.ones((8, 2))
        axes, _ = heatmap.directions(points, 1e-8)
        result = heatmap.train_trial(
            features,
            points,
            offset,
            scale,
            points,
            axes,
            targets,
            indices,
            config,
            "test",
            17,
            0,
            tmp_path,
            device,
            {},
        )
        head = heatmap.HeatmapHead(8, config["head"])
        head.load_state_dict(result["state"])
        coords, order = heatmap.predict(
            head,
            heatmap.loader(features, points, indices["test"], config),
            device,
            config,
        )
        predictions = heatmap.evaluate_coords(
            coords, order, offset, scale, points, axes, targets, config
        )["angles"]
    else:
        monkeypatch.setattr(spatial, "ROOT", tmp_path)
        lines = np.array(
            [
                [[0.2, 0.2], [0.8, 0.8]],
                [[0.3, 0.1], [0.5, 0.9]],
                [[0.7, 0.1], [0.7, 0.9]],
            ],
            dtype=np.float32,
        )
        records = [
            spatial.HVRecord(str(i), "", i, 16, 16, "left", lines, targets[i], split)
            for split, rows in indices.items()
            for i in rows
        ]
        splits = {split: [records[i] for i in rows] for split, rows in indices.items()}
        monkeypatch.setattr(spatial, "dense_feature_cache", lambda *a: (features, {}))
        monkeypatch.setattr(spatial, "line_target_cache", lambda *a: (masks, {}))
        config.update(
            lines=dict(
                config["head"],
                **config["mask_loss"],
                mask_threshold=0.5,
                axis_probability_power=4.0,
                axis_probability_quantile=0.95,
                minimum_hard_axis_pixels=5,
            ),
            seeds=[17],
            num_workers=0,
        )
        result = spatial.train_line_model(
            "test",
            dict(dense_channels=8, display_name="Test"),
            splits,
            config,
            tmp_path,
            device,
        )
        table = pd.read_csv(tmp_path / "line_predictions_test.csv")
        assert result["status"] == "complete" and len(table) == 2
        assert (tmp_path / "heads/lines/test/seed17.pt").exists()
        assert np.isfinite(table.select_dtypes(include=np.number)).all().all()
        return
    assert result["best_epoch"] == 1
    assert predictions.shape == (2, 2) and np.isfinite(predictions).all()
    np.testing.assert_array_equal(order, indices["test"])


def test_pooled_default_config_loads_patient_split(tmp_path, monkeypatch):
    """Exercise the shared loader with the released P-dir configuration."""
    cfg = json.loads(pooled.DEFAULT_CONFIG.read_text())
    cfg["angle_result_manifest"] = "split.json"
    image_dir = tmp_path / "data/images"
    image_dir.mkdir(parents=True)
    image_file = tmp_path / "sample.png"
    Image.new("RGB", (16, 32)).save(image_file)
    rows = []
    splits = {}
    patient_start = 0
    for split, n_images, n_patients in [
        ("train", 829, 617),
        ("val", 172, 132),
        ("test", 176, 133),
    ]:
        splits[split] = []
        for i in range(n_images):
            name = f"{split}_{i}.png"
            (image_dir / name).hardlink_to(image_file)
            splits[split].append(name)
            rows.append(
                dict(
                    filename=name,
                    patient_id=patient_start + i % n_patients,
                    image_width=16,
                    image_height=32,
                    labels="left",
                    HVA=20.0,
                    IMA=10.0,
                    great_toe=".2,.1,.8,.9",
                    first_metatarsal=".3,.1,.5,.9",
                    second_metatarsal=".7,.1,.7,.9",
                )
            )
        patient_start += n_patients
    dataset = tmp_path / "data/datasets.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    (tmp_path / "split.json").write_text(json.dumps({"split_ids": splits}))
    monkeypatch.setattr(spatial, "ROOT", tmp_path)
    monkeypatch.setattr(spatial, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(spatial, "DATASET_CSV", dataset)
    loaded, audit = spatial.load_locked_records(cfg)
    assert [len(loaded[s]) for s in splits] == [829, 172, 176]
    assert audit["split_patient_counts"] == dict(train=617, val=132, test=133)

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/benchmarks"))
import evaluate as evaluation
import hvangle_dense_controls_benchmark as controls
import hvangle_dense_direct_benchmark as direct
import hvangle_dense_heatmap_benchmark as heatmap
import hvangle_frozen_spatial_benchmark as spatial
import hvangle_pooled_angle_replay as pooled

SETTINGS = dict(
    canonical_feature_grid=14,
    projection_channels=256,
    decoder_channels=[128, 64, 32, 16],
)


@pytest.mark.parametrize(
    "channels,mask_count,direct_count", [(768, 589331, 595554), (2048, 917011, 923234)]
)
def test_paper_parameter_budgets(channels, mask_count, direct_count):
    count = lambda model: sum(p.numel() for p in model.parameters())
    assert count(spatial.LineDecoder(channels, SETTINGS)) == mask_count
    assert count(direct.DenseAngleHead(channels, SETTINGS)) == direct_count
    assert count(controls.GridDirectHead(channels, SETTINGS)) == direct_count
    assert count(controls.AuxiliaryMaskHead(channels, SETTINGS)) == direct_count + 51
    assert count(heatmap.HeatmapHead(channels, SETTINGS)) == mask_count + 51


def test_pooled_parameter_budget():
    assert sum(p.numel() for p in pooled.AngleHead(512, 256).parameters()) == 131842
    assert sum(p.numel() for p in pooled.AngleHead(1024, 512).parameters()) == 525826


def test_unoriented_angle_and_reference_substitution():
    radians = np.deg2rad([30, 10, 0])
    axes = np.stack([np.cos(radians), np.sin(radians)], axis=-1)
    np.testing.assert_allclose(evaluation.measure(axes), [20, 10], atol=1e-12)
    axes[1] *= -1
    np.testing.assert_allclose(evaluation.measure(axes), [20, 10], atol=1e-12)
    axes[0] = axes[1]
    np.testing.assert_allclose(evaluation.measure(axes), [0, 10], atol=1e-12)


def test_pca_original_coordinate_aspect_ratio_and_fallback():
    probability = np.eye(20) * 0.9
    hard = probability >= 0.5
    _, direction, fallback = spatial.probability_axis(
        probability, 200, 100, 4, 0.95, hard, 5
    )
    expected = np.array([2, 1]) / np.sqrt(5)
    assert abs(direction @ expected) == pytest.approx(1)
    assert not fallback
    _, soft_direction, fallback = spatial.probability_axis(
        probability, 200, 100, 4, 0.95, np.zeros_like(hard), 5
    )
    assert fallback
    assert abs(soft_direction @ expected) == pytest.approx(1)


def test_endpoint_loss_is_invariant_to_endpoint_order():
    torch.manual_seed(17)
    logits = torch.randn(2, 6, 16, 16, requires_grad=True)
    points = torch.rand(2, 3, 2, 2) * 16
    settings = dict(temperature=1, sigma_px=2)
    original = heatmap.endpoint_loss(logits, points, settings)
    swapped = heatmap.endpoint_loss(logits, points.flip(2), settings)
    torch.testing.assert_close(original, swapped)
    original.backward()
    assert torch.isfinite(logits.grad).all()


def test_patient_cluster_resampling_preserves_image_weighting():
    differences = np.array([[1.0, 4.0], [1.0, 4.0], [9.0, 2.0]])
    patients = np.array([10, 10, 2])
    actual = evaluation.bootstrap(differences, patients)
    # Two clusters contain 1 and 2 images: mixed draws average to 11/3, not 5.
    draws = np.random.default_rng(20260826).integers(2, size=(5000, 2))
    counts = np.array([1, 2])
    sums = np.array([[9.0, 2.0], [2.0, 8.0]])
    expected = sums[draws].sum(1) / counts[draws].sum(1)[:, None]
    np.testing.assert_allclose(
        actual, np.quantile(expected, [0.025, 0.975, 0.05 / 24, 1 - 0.05 / 24], axis=0)
    )


def test_prediction_rows_reject_duplicates():
    import pandas as pd

    frame = pd.DataFrame(
        dict(model_id=["x"] * 6, seed=[17, 17, 42, 42, 73, 73], identifier=["a"] * 6)
    )
    with pytest.raises(ValueError, match="identifiers"):
        evaluation.load_seed_frames(frame, "x", ["a"])

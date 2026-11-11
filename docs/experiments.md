# Training and Evaluation

Complete [data preparation](data.md) first. Run commands from the repository root. CUDA is the default; use `--device cpu` for CPU execution.

## 1. M-mask and Feature Extraction

```bash
# Five medical encoders
python scripts/benchmarks/hvangle_frozen_spatial_benchmark.py --run

# ImageNet-RN50
python scripts/benchmarks/hvangle_frozen_spatial_benchmark.py \
  --config scripts/benchmarks/hvangle_imagenet_config.json \
  --output outputs/benchmarks/controlled_cross_model_v2/medical/hvangle_spatial_imagenet_rn50 \
  --run
```

These commands generate pooled features, spatial features, mask targets, and M-mask predictions. Both output roots are required before running the default six-encoder comparisons below.

Use `--models biomedclip` to select one encoder, `--audit-only` to check the data and targets, or `--extract-only` to generate features without training a head.

## 2. Additional Readouts

```bash
# G-dir and A-dir
python scripts/benchmarks/hvangle_dense_controls_benchmark.py --run

# D-dir
python scripts/benchmarks/hvangle_dense_direct_benchmark.py --run

# P-dir
python scripts/benchmarks/hvangle_pooled_angle_replay.py --run

# H-end
python scripts/benchmarks/hvangle_dense_heatmap_benchmark.py --run
```

| Readout | Input and prediction | Checkpoint selection |
|:--|:--|:--|
| M-mask | Spatial features → three masks → PCA axes → angles | Validation case-macro Dice; loss breaks ties |
| G-dir | Spatial-grid convolutions → angles | Validation joint angle MAE |
| D-dir | Upsampling decoder → spatial pooling → angles | Validation joint angle MAE |
| A-dir | D-dir with auxiliary mask supervision | Validation joint angle MAE |
| P-dir | Pooled features → MLP → angles | Validation joint angle MAE |
| H-end | Six unordered endpoint heatmaps → axes → angles | Validation joint angle MAE |

All routes use seeds 17/42/73, batch size 16, at most 60 epochs, patience 10, and three AdamW learning-rate/weight-decay candidates. Direct heads use Huber loss with δ = 5°. M-mask uses weighted BCE and soft Dice (0.5 each); A-dir adds this mask loss with weight 1. P-dir uses neither AMP nor gradient clipping. H-end uses endpoint-order-invariant Gaussian cross-entropy.

Mask threshold is 0.5. PCA uses original image coordinates; fewer than five positive pixels triggers the top-5%-probability fallback with fourth-power weights. Class weights use training masks only.

**Numerical precision:** Mask-loss sigmoid and reductions use float32 under AMP to avoid float16 overflow. This fixes the historical M-mask implementation, so retraining may differ from the paper checkpoints. Evaluation of saved predictions is unchanged.

G-dir/A-dir, D-dir, and H-end also support `--audit-only` and `--smoke-test-heads` after feature preparation. All training scripts accept `--models biomedclip` for a single-encoder run. Checkpoints and predictions are saved under the configured output directories.

## 3. Evaluation

```bash
python scripts/benchmarks/evaluate.py \
  --mask-roots outputs/benchmarks/controlled_cross_model_v2/medical/hvangle_spatial \
               outputs/benchmarks/controlled_cross_model_v2/medical/hvangle_spatial_imagenet_rn50 \
  --direct G-dir=outputs/benchmarks/controlled_cross_model_v2/medical/hvangle_dense_review_controls_v1/controls_predictions.csv \
  --output outputs/evaluation
```

Repeat `--direct ROUTE=CSV` for other readouts. Control tables are filtered automatically: G-dir selects `grid_direct`, and A-dir selects `aux_mask`. Each supplied table must cover every evaluated encoder, test image, and seed exactly once. Concatenate per-run prediction CSVs if needed. Missing or duplicate rows, inconsistent references, and non-finite predictions are rejected.

| Output | Contents |
|:--|:--|
| `recovery.csv` | Per-axis and macro axis MAE |
| `measurement.csv` | Angle MAE and sample SD across seeds |
| `contrasts.csv` | Paired M-mask − direct differences and confidence intervals |
| `substitution.csv` | One-axis reference-substitution gains and exploratory intervals |

Contrasts average per-image absolute-error differences across seeds before patient-cluster bootstrap resampling: 5,000 draws, seed 20260826, numeric patient ordering, and image-weighted cluster means. Intervals are ordinary 95% and Bonferroni-adjusted for 12 comparisons per comparator (six encoders × two angles). Positive differences favor direct regression. Positive substitution gains indicate reduced error after reference replacement; substitution is a diagnostic requiring reference annotations.

## Tests

```bash
pip install pytest
pytest -q
```

Tests cover geometry, parameter counts, half-precision loss stability, data loading, prediction validation, patient-cluster resampling, and a small training/checkpoint round trip for each readout. Full six-encoder retraining is not part of the test suite.

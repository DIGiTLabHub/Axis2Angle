# Training and evaluation

Run commands from the repository root after [data preparation](data.md). CUDA is the default; `--device cpu` selects CPU explicitly. Each route searches the same three learning-rate/weight-decay candidates using validation data, with seeds 17/42/73, batch size 16, at most 60 epochs, and patience 10.

## M-mask and feature extraction

```bash
python scripts/benchmarks/hvangle_frozen_spatial_benchmark.py --audit-only
python scripts/benchmarks/hvangle_frozen_spatial_benchmark.py --run
python scripts/benchmarks/hvangle_frozen_spatial_benchmark.py \
  --config scripts/benchmarks/hvangle_imagenet_config.json \
  --output outputs/benchmarks/controlled_cross_model_v2/medical/hvangle_spatial_imagenet_rn50 --run
```

`--models biomedclip` selects one encoder. `--extract-only` generates pooled/dense feature caches without fitting a head. The five medical encoders and ImageNet use separate output roots, matching the downstream configurations.

M-mask uses weighted BCE + soft Dice (0.5 each), selects checkpoints by validation case-macro Dice, and fits PCA directions after mapping masks back to original image coordinates. Mask threshold is 0.5; fewer than five positive pixels triggers the top-5%-probability, fourth-power-weighted PCA fallback. Positive class weights use training masks only.

## Direct and alternative readouts

```bash
# G-dir: spatial-grid direct regression; A-dir: direct regression with auxiliary masks.
python scripts/benchmarks/hvangle_dense_controls_benchmark.py --run

# D-dir: upsampling decoder followed by pooling and angle regression.
python scripts/benchmarks/hvangle_dense_direct_benchmark.py --run

# P-dir: pooled feature vector and MLP.
python scripts/benchmarks/hvangle_pooled_angle_replay.py --run

# H-end: six endpoint heatmaps and the fixed angle protocol.
python scripts/benchmarks/hvangle_dense_heatmap_benchmark.py --run
```

G-dir, D-dir, and A-dir use Huber loss with δ = 5° and validation joint HVA/IMA MAE for selection. A-dir adds the fixed-weight auxiliary mask loss. P-dir uses the same angle loss without AMP or gradient clipping. H-end uses unordered endpoint supervision and validation angle MAE. Encoders are never fine-tuned. Direct and mask routes retain their distinct training objectives and checkpoint criteria from the paper.

The dense scripts also support `--audit-only`, `--smoke-test-heads`, and `--models biomedclip`. Their input audits require prepared data and feature caches. Selected checkpoints and predictions are written under each route's configured output directory.

## RQ1–RQ3 analysis

```bash
python scripts/benchmarks/evaluate.py \
  --mask-roots outputs/benchmarks/controlled_cross_model_v2/medical/hvangle_spatial \
               outputs/benchmarks/controlled_cross_model_v2/medical/hvangle_spatial_imagenet_rn50 \
  --direct G-dir=outputs/benchmarks/controlled_cross_model_v2/medical/hvangle_dense_review_controls_v1/controls_predictions.csv \
  --output outputs/evaluation
```

For control tables, the evaluator selects `grid_direct` for G-dir and `aux_mask` for A-dir automatically. Concatenate per-model/seed tables if supplying selected-run CSVs instead. Every `--direct ROUTE=CSV` must contain exactly one prediction per test image and seed for every evaluated encoder, with the reference angle and patient columns emitted by the training scripts. Duplicate rows, missing images, or inconsistent references are rejected. Repeat `--direct` for additional routes.

Outputs contain axis recovery, angle MAE, paired route contrasts, and one-axis reference substitutions. Contrasts average absolute-error differences across seeds per image, then use 5,000 patient-cluster bootstrap draws, seed 20260826, numeric patient ordering, and image-weighted cluster resampling. They report ordinary 95% and Bonferroni-adjusted intervals for a 12-comparison family (six encoders × two angles) per comparator. Positive `M-mask − direct` differences favor direct regression. Reference-substitution gains are exploratory, with positive values indicating lower error after replacement; they are not deployable predictions.

## Code checks

```bash
pip install pytest
pytest -q
```

Tests exercise head sizes, geometry conventions, unordered endpoint loss, and patient-cluster resampling. They do not replace the full six-encoder training experiment.

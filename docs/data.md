# Data Preparation

Run all commands from the repository root.

## HVAngleEst

Obtain HVAngleEst from its authors and keep the released structure:

```text
HVAngleEst/
├── HVAngleEst/
│   ├── datasets.csv
│   └── images/
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

`datasets.csv` contains filenames, patient IDs, HVA/IMA, image dimensions, laterality, and three normalized endpoint pairs: `great_toe` (PH1), `first_metatarsal` (M1), and `second_metatarsal` (M2). Images and annotations are not included in this repository.

```bash
python scripts/benchmarks/prepare_split.py
```

The split retains unilateral images from the released image pool. Sorted patient IDs are shuffled with NumPy `RandomState(42)` and assigned 70/15/15; image order follows the annotation CSV.

| Split | Images | Patients |
|:--|--:|--:|
| Train | 829 | 617 |
| Validation | 172 | 132 |
| Test | 176 | 133 |

Patients do not overlap across splits. The script rejects a conflicting existing split.

## Pretrained Encoders

Checkpoint paths and SHA-256 hashes are specified in the JSON files under `scripts/benchmarks/`. Place the original weights at these paths or update the paths to copies inside the repository. External files can be linked into the repository. Model files are verified before extraction and are not downloaded automatically.

| Encoder | Required files | Configuration |
|:--|:--|:--|
| BiomedCLIP | Original checkpoint and `open_clip_config.json`; loaded with `open_clip` | `hvangle_frozen_spatial_config.json` |
| MedImageInsight | Original `MedImageInsights/medimageinsightmodel.py` implementation and `2024.09.27/` model directory | `hvangle_frozen_spatial_config.json` |
| MedCLIP-RN50 / Swin | Checkpoints in `checkpoints/MedCLIP/`; Swin also requires the original backbone configuration | `hvangle_frozen_spatial_config.json` |
| DAX-RN50-vA | DAX teacher checkpoint `checkpoints/DAX/dax-resnet50-a.pth` | `hvangle_frozen_spatial_config.json` |
| ImageNet-RN50 | Torchvision checkpoint `resnet50-0676ba61.pth` | `hvangle_imagenet_config.json` |

All encoders remain frozen. MedImageInsight uses its internal final pre-pooling grid and the original JPEG95 preprocessing. Feature caches record annotation, split, image-order, and checkpoint hashes; regenerate them after changing these inputs.

Continue with [training and evaluation](experiments.md).

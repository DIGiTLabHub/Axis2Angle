# Data and frozen encoders

Obtain HVAngleEst from its authors and preserve the released directory structure:

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

The CSV supplies filenames, patient IDs, HVA/IMA, original dimensions, laterality, boxes, and three normalized endpoint pairs (`great_toe`, `first_metatarsal`, `second_metatarsal`). `great_toe` denotes the PH1 axis. Dataset images and annotations are not distributed here.

```bash
python scripts/benchmarks/prepare_split.py
```

This retains unilateral images in the released image pool and partitions sorted patient IDs using NumPy `RandomState(42)` with 70/15/15 allocation. Image order follows the annotation CSV. It reproduces 829/172/176 images from 617/132/133 disjoint patients. An existing different split is rejected.

## Encoder files

The JSON configurations in `scripts/benchmarks/` specify checkpoint paths and SHA-256 hashes used in the experiments. Obtain the original pretrained weights and preserve these paths, or edit the paths to your local copies. Model files are checked before extraction; downloads do not occur during training.

| Encoder | Required source or package | Configuration |
|:--|:--|:--|
| BiomedCLIP | `open_clip`; original `open_clip_config.json` and checkpoint | `hvangle_frozen_spatial_config.json` |
| MedImageInsight | Original `MedImageInsights/medimageinsightmodel.py` implementation and `2024.09.27/` model directory | `hvangle_frozen_spatial_config.json` |
| MedCLIP-RN50 / Swin | Original MedCLIP source at `external_models/MedCLIP/`; checkpoints at `checkpoints/MedCLIP/` | `hvangle_frozen_spatial_config.json` |
| DAX-RN50-vA | DAX teacher checkpoint at `checkpoints/DAX/dax-resnet50-a.pth` | `hvangle_frozen_spatial_config.json` |
| ImageNet-RN50 | Torchvision ResNet-50 checkpoint `resnet50-0676ba61.pth` | `hvangle_imagenet_config.json` |

MedImageInsight uses its internal final pre-pooling token grid. The public pooled embedding alone cannot supply the spatial readouts. Its original JPEG95 preprocessing is retained. All encoder weights remain frozen. Features and targets are cached with hashes of the image order, annotations, split, and model files. Regenerate caches after changing these inputs.

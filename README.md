<div align="center">

<h1>Does Predicting Anatomical Axes Improve Foot Angle Measurement?<br><sub>Recovery, Measurement Utility, and Reference-Axis Substitution</sub></h1>

**Hongcheng Jiang &nbsp; Cheng Han\* &nbsp; ZhiQiang Chen &nbsp; Dianxiang Xu**

University of Missouri–Kansas City, United States<br>
<sup>*</sup>Corresponding author

[**Overview**](#overview) · [**Results**](#main-results) · [**Getting Started**](#getting-started) · [**Citation**](#citation)

</div>

## Overview

Official implementation of **Axis2Angle**, a six-encoder study of anatomical-axis recovery, angle measurement, and reference-axis substitution.

<p align="center">
  <img src="assets/framework-layout.png" width="100%" alt="Axis2Angle framework">
</p>

**Figure 1.** Both routes use the same frozen spatial features. The direct head predicts HVA and IMA; the mask head exposes PH1/M1/M2 axes through PCA and applies the fixed measurement protocol. RQ1 evaluates axis recovery, RQ2 compares angle accuracy, and RQ3 substitutes reference axes to isolate their effects.

## Main Results

**HVAngleEst:** 176 test images from 133 patients, patient-disjoint split. MAE (°), mean ± sample SD over three seeds. **G-dir:** spatial direct regression. **M-mask:** axis masks followed by the fixed angle protocol.

| Frozen encoder | Axis MAE | G-dir HVA | M-mask HVA | G-dir IMA | M-mask IMA |
|:--|--:|--:|--:|--:|--:|
| BiomedCLIP | 2.570 ± 0.145 | 5.269 ± 0.103 | 4.166 ± 0.136 | 2.676 ± 0.090 | 2.726 ± 0.022 |
| MedImageInsight | 2.473 ± 0.027 | 4.054 ± 0.033 | 3.746 ± 0.160 | 2.255 ± 0.007 | 2.318 ± 0.100 |
| MedCLIP-RN50 | 4.032 ± 0.202 | 5.544 ± 0.093 | 5.463 ± 0.421 | 2.571 ± 0.043 | 3.918 ± 0.108 |
| MedCLIP-Swin | 4.407 ± 0.369 | 5.458 ± 0.257 | 6.032 ± 0.408 | 2.743 ± 0.111 | 4.111 ± 0.249 |
| DAX-RN50-vA | 6.779 ± 0.409 | 7.992 ± 0.090 | 10.302 ± 0.536 | 2.934 ± 0.083 | 5.709 ± 0.925 |
| ImageNet-RN50 | 4.468 ± 0.292 | 6.052 ± 0.265 | 6.192 ± 0.079 | 2.811 ± 0.149 | 3.665 ± 0.539 |

Axis recovery and angle accuracy must be evaluated separately: adjusted intervals support no HVA improvement over G-dir and higher IMA error for four of six encoders.

<details>
<summary><b>Reference-axis substitution</b></summary>

<p align="center">
  <img src="assets/reference_axis_gains.png" width="100%" alt="Reference-axis substitution results">
</p>

**Paper figure.** Replacing MedImageInsight's predicted PH1 axis with its reference lowers HVA MAE by 45.0%. The shared M1 axis affects HVA and IMA differently. Substitution uses reference annotations and serves as a diagnostic.

</details>

## Getting Started

### Installation

Python 3.11 · PyTorch 2.4

```bash
git clone https://github.com/DIGiTLabHub/Axis2Angle.git
cd Axis2Angle
pip install -r requirements.txt
```

### Data Preparation

Follow [data and encoder setup](docs/data.md) to obtain HVAngleEst and the pretrained encoders, then recreate the patient-disjoint split:

```bash
python scripts/benchmarks/prepare_split.py
```

### Training and Evaluation

Example with BiomedCLIP:

```bash
# M-mask: frozen feature extraction and axis-mask training
python scripts/benchmarks/hvangle_frozen_spatial_benchmark.py --models biomedclip --run

# G-dir and A-dir: spatial direct regression and auxiliary mask supervision
python scripts/benchmarks/hvangle_dense_controls_benchmark.py --models biomedclip --run
```

See [training and evaluation](docs/experiments.md) for all six encoders, P-dir, D-dir, H-end, paired comparisons, and reference-axis substitution.

## Citation

```bibtex
@inproceedings{jiang2027axis2angle,
  title={Does Predicting Anatomical Axes Improve Foot Angle Measurement? Recovery, Measurement Utility, and Reference-Axis Substitution},
  author={Jiang, Hongcheng and Han, Cheng and Chen, ZhiQiang and Xu, Dianxiang},
  booktitle={WACV Workshop on Latent Visual Reasoning (LVR)},
  year={2027},
  url={https://github.com/DIGiTLabHub/Axis2Angle}
}
```

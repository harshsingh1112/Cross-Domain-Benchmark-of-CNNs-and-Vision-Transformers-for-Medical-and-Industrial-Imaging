# Fixed-Budget Cross-Domain Vision Benchmark

## Overview

This project implements a controlled benchmark to compare three widely used vision model families - **ResNet-50**, **EfficientNet-B0**, and **ViT-Tiny** across two applied image classification domains:

- **Medical imaging**
- **Industrial / applied science imaging**

The goal is **not** to achieve state-of-the-art results.  
Instead, the focus is on understanding how different model families behave **under identical training constraints** when the application domain changes.

All experiments follow a fixed setup (image resolution, optimizer, epochs, etc.) to ensure fairness and reproducibility.

---

## Research Motivation

CNNs and Vision Transformers are often evaluated within a single domain.  
However, model performance and efficiency can vary significantly when the visual characteristics of the data change.

This project aims to:
- Compare CNN and Transformer model families under a fixed compute budget
- Study cross-domain behavior rather than single-dataset performance
- Provide a clean, reproducible benchmarking pipeline suitable for academic use

---

## Domains and Sub-Domains

### Medical Domain

**Experimentally evaluated sub-domain**
- Histopathology image classification  
  - Dataset: **PathMNIST**

**Contextual (literature-based) sub-domains**
1. Chest X-ray disease classification  
2. Brain MRI tumor classification  
3. Skin lesion (dermatology) classification  
4. Retinal fundus disease detection  
5. OCT retinal imaging  
6. Ultrasound image classification  
7. Mammography breast cancer screening  
8. Bone fracture X-ray classification  
9. Organ classification in CT/MRI  

These sub-domains are discussed in the `analysis/` folder and are **not experimentally evaluated** in this codebase.

---

### Industrial / Applied Science Domain

**Experimentally evaluated sub-domain**
- Remote sensing / aerial image classification  
  - Dataset: **EuroSAT**

**Contextual (literature-based) sub-domains**
1. Aerial UAV scene classification  
2. Astronomy image classification (galaxies, stars)  
3. Climate and weather pattern imaging  
4. Energy infrastructure monitoring  
5. Materials science microstructure classification  
6. Aerospace structural inspection  
7. Geophysical / seismic image classification  
8. Optical physics / microscopy (non-medical)  
9. Fluid dynamics flow visualization  

These are included for interpretation and discussion only.

---

## Models Benchmarked

1. **ResNet-50** – Classical CNN baseline  
2. **EfficientNet-B0** – Parameter-efficient CNN  
3. **ViT-Tiny** – Transformer-based vision model (via `timm`)

---

## Project Structure

```
vision-benchmark/
├── configs/
├── datasets/
├── models/
├── analysis/
├── results/
├── train.py
└── evaluate.py
```

---

## Execution Instructions

### macOS (Apple Silicon)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py --config configs/pathmnist_resnet.yaml --epochs 2
```

### Windows

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py --config configs/pathmnist_resnet.yaml --epochs 2
```

### Google Colab (Recommended)

```python
!git clone https://github.com/<your-username>/vision-benchmark.git
%cd vision-benchmark
!pip install -r requirements.txt

!python train.py --config configs/pathmnist_resnet.yaml
!python train.py --config configs/pathmnist_efficientnet.yaml
!python train.py --config configs/pathmnist_vit.yaml
!python train.py --config configs/eurosat_resnet.yaml
!python train.py --config configs/eurosat_efficientnet.yaml
!python train.py --config configs/eurosat_vit.yaml
```

---

## Generating Results

```bash
python evaluate.py --plot-only
```

Outputs:
- `results/final_results.csv`
- `results/plots/`

---

## Notes

- Only PathMNIST and EuroSAT are experimentally evaluated.
- Other sub-domains are literature-backed.
- Results must be interpreted within this fixed-budget setup.

---

## License

Academic and educational use only.

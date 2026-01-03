# Fixed-Budget Cross-Domain Vision Benchmark

## Project Overview
This project benchmarks three vision model families (ResNet-50, EfficientNet-B0, ViT-Tiny) across two distinct domains: **Medical** (Pathology) and **Industrial** (Remote Sensing). The goal is to evaluate performance and efficiency under strict, fixed experimental constraints.

## Scope & Limitations
- **Experimentally Evaluated**: 
  - Medical: PathMNIST (Histopathology)
  - Industrial: EuroSAT (Land Use/Cover)
- **Literature-Inferred**: 18 other sub-domains (9 per domain) are analyzed contextually in the `analysis/` folder but are not experimentally validated in this codebase.

## Directory Structure
```
vision-benchmark/
├── configs/        # YAML configuration files for experiments
├── datasets/       # Data loaders for PathMNIST and EuroSAT
├── models/         # Wrappers for ResNet, EfficientNet, ViT
├── analysis/       # Markdown templates for domain analysis
├── results/        # Output logs, CSVs, and plots
├── train.py        # Main training script
└── evaluate.py     # Plotting and evaluation script
```

## How to Run Locally (Mac M1/Apple Silicon)

1. **Prerequisites**: Ensure you have Python 3.9+ installed.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On Mac M1, PyTorch uses MPS acceleration automatically if available (handled in `train.py`).*

3. **Run Experiments**:
   Run the training script with a specific config file:
   ```bash
   # Example: Train ResNet on PathMNIST
   python train.py --config configs/pathmnist_resnet.yaml
   
   # Example: Train ViT on EuroSAT
   python train.py --config configs/eurosat_vit.yaml
   ```

## How to Run on Google Colab

1. **Upload Code**: Zip the `vision-benchmark` folder and upload it to your Google Drive.
2. **Mount Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/vision-benchmark
   ```
3. **Install Requirements**:
   ```python
   !pip install -r requirements.txt
   ```
4. **Run Training**:
   ```python
   # Run all experiments sequentially
   !python train.py --config configs/pathmnist_resnet.yaml
   !python train.py --config configs/pathmnist_efficientnet.yaml
   !python train.py --config configs/pathmnist_vit.yaml
   
   !python train.py --config configs/eurosat_resnet.yaml
   !python train.py --config configs/eurosat_efficientnet.yaml
   !python train.py --config configs/eurosat_vit.yaml
   ```
5. **Generate Plots**:
   ```python
   !python evaluate.py --plot-only
   ```
   Plots will be saved in `results/plots/`.

## Generating Research Artifacts
1. **Tables**: The file `results/final_results.csv` contains all raw metrics. Import this into Excel/LaTeX to generate your comparison tables.
2. **Plots**: Run `python evaluate.py --plot-only` to generate:
   - `accuracy_comparison.png`
   - `accuracy_vs_efficiency.png`

# CNN Robustness Evaluation on CIFAR-10

Capstone project evaluating the corruption robustness of three CNN architectures under three training strategies on the CIFAR-10 dataset.

---

## Project Objective

We train three CNN architectures (BaselineCNN, LightweightResNet, WiderCNN) under three training conditions:

- **Baseline** — standard training, no augmentation beyond normalization
- **Standard Augmentation (StdAug)** — random horizontal flip and random crop
- **Degradation-Aware Augmentation (DegAug)** — StdAug plus random corruption applied during training

Each trained model is evaluated on 25 corruption conditions (5 corruption types × 5 severities) derived from five synthetic corruptions: Gaussian Noise, Gaussian Blur, JPEG Compression, Resolution Reduction, and Brightness Change.

---

## Repository Structure

```
Capstone/
├── run_all.py                      # Full pipeline entry point
├── requirements.txt
├── data/                           # CIFAR-10 (auto-downloaded, git-ignored)
├── models/                         # Saved .pth weight files (git-ignored)
├── results/                        # CSVs and figures (git-ignored)
│   ├── figures/
│   │   └── poster/                 # Poster-ready PNG figures
│   └── *.csv
└── src/
    ├── data_loader.py
    ├── corruption_transforms.py
    ├── models/
    │   ├── baseline_cnn.py
    │   ├── resnet_cnn.py
    │   └── wider_cnn.py
    ├── train_baseline_cnn.py
    ├── train_resnet.py
    ├── train_wider_cnn.py
    ├── train_baseline_cnn_stdaug.py
    ├── train_resnet_stdaug.py
    ├── train_wider_cnn_stdaug.py
    ├── train_baseline_cnn_degaug.py
    ├── train_resnet_degaug.py
    ├── train_wider_cnn_degaug.py
    ├── evaluate_robustness.py
    ├── visualize_robustness.py
    ├── generate_poster_plots.py
    ├── compute_poster_stats.py
    └── average_runs.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically on first run into `data/`.

---

## How to Run

### Full pipeline (train + evaluate + visualize)

```bash
python run_all.py
```

All 9 models are trained with `--seed 42`. Training 9 models takes several hours on CPU; use a GPU where possible.

### Skip training (use existing weight files)

```bash
python run_all.py --skip-training
```

Requires pre-trained `.pth` files in `models/`.

### Train a single model

```bash
python src/train_baseline_cnn.py --seed 42
python src/train_resnet_stdaug.py --seed 42
python src/train_wider_cnn_degaug.py --seed 42
```

### Evaluate a single model

```bash
python src/evaluate_robustness.py \
    --model baseline_cnn \
    --weights models/baseline_cnn_seed42_best.pth \
    --output results/baseline_cnn_robustness_seed42.csv
```

`--model` accepts: `baseline_cnn`, `resnet`, `wider_cnn`

### Generate figures

```bash
python src/visualize_robustness.py          # 21 analysis figures
python src/generate_poster_plots.py         # 3 poster figures
```

### Compute summary statistics

```bash
python src/compute_poster_stats.py
```

Prints two tables and saves `results/poster_strategy_comparison.csv` and `results/wider_cnn_baseline_heatmap_data.csv`.

### Average metrics across seeds

```bash
python src/average_runs.py
```

Reads training metric CSVs for seeds 42, 123, 456 per baseline model and outputs per-epoch mean ± std accuracy to `results/{model}_averaged_metrics.csv`.

---

## Outputs

| Location | Contents |
|---|---|
| `models/*.pth` | Best and final weights for each of the 9 models |
| `results/*_metrics.csv` | Per-epoch train loss, test loss, accuracy |
| `results/*_robustness.csv` | Per-corruption accuracy (25 conditions + clean) |
| `results/figures/*.png` | 21 robustness analysis plots |
| `results/figures/poster/*.png` | 3 publication-ready poster figures |
| `results/poster_strategy_comparison.csv` | Mean accuracy table (corruption × strategy) |
| `results/wider_cnn_baseline_heatmap_data.csv` | WiderCNN baseline accuracy grid |

---

## How Figures in the Paper Were Generated

All figures in the paper were generated from robustness CSV files produced by `evaluate_robustness.py`.

**Poster figures** (`plot1_corruption_overview.png`, `plot2_model_strategy.png`, `plot3_strategy_per_corruption.png`) were produced by:

```bash
python src/generate_poster_plots.py
```

Values are computed dynamically from the 9 robustness CSV files. The color scheme uses `#B0BEC5` (Baseline), `#4A6FA5` (Std Aug), and `#B8940F` (Deg Aug).

**Analysis figures** (line plots, bar charts, heatmaps) were produced by:

```bash
python src/visualize_robustness.py
```

To fully reproduce all figures from scratch:

```bash
python run_all.py
```

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

CSV_FILES = {
    "baseline_cnn": RESULTS_DIR / "baseline_cnn_robustness_seed42.csv",
    "resnet":       RESULTS_DIR / "resnet_robustness_seed123.csv",
    "wider_cnn":    RESULTS_DIR / "wider_cnn_robustness_seed456.csv",
}

COLORS = {
    "baseline_cnn": "blue",
    "resnet":       "orange",
    "wider_cnn":    "green",
}

LABELS = {
    "baseline_cnn": "BaselineCNN",
    "resnet":       "LightweightResNet",
    "wider_cnn":    "WiderCNN",
}

CORRUPTION_TITLES = {
    "gaussian_noise":       "Gaussian Noise",
    "gaussian_blur":        "Gaussian Blur",
    "jpeg_compression":     "JPEG Compression",
    "resolution_reduction": "Resolution Reduction",
    "brightness":           "Brightness Change",
}


def load_csv(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["corruption_type"], int(row["severity"]))
            rows[key] = float(row["accuracy"])
    return rows


def plot_corruption_lines(data, corruption_type, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, rows in data.items():
        severities = [1, 2, 3, 4, 5]
        accuracies = [rows[(corruption_type, s)] for s in severities]
        ax.plot(severities, accuracies, marker="o", color=COLORS[model], label=LABELS[model])
    ax.set_title(CORRUPTION_TITLES[corruption_type])
    ax.set_xlabel("Severity")
    ax.set_ylabel("Accuracy")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_clean_bar(data, out_path):
    models = list(data.keys())
    accuracies = [data[m][("clean", 0)] for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(x, accuracies, color=[COLORS[m] for m in models], width=0.5)
    ax.set_title("Clean Accuracy")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.grid(True, axis="y")
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{acc:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(rows, model, out_path):
    corruption_types = list(CORRUPTION_TITLES.keys())
    severities = [1, 2, 3, 4, 5]
    matrix = np.array([
        [rows[(ct, s)] for s in severities]
        for ct in corruption_types
    ])
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_title(f"{LABELS[model]} — Corruption Accuracy")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Corruption Type")
    ax.set_xticks(range(len(severities)))
    ax.set_xticklabels(severities)
    ax.set_yticks(range(len(corruption_types)))
    ax.set_yticklabels([CORRUPTION_TITLES[ct] for ct in corruption_types])
    for i in range(len(corruption_types)):
        for j in range(len(severities)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = {model: load_csv(path) for model, path in CSV_FILES.items()}

    for corruption_type in CORRUPTION_TITLES:
        out = FIGURES_DIR / f"corruption_{corruption_type}.png"
        plot_corruption_lines(data, corruption_type, out)
        print(f"Saved {out.name}")

    out = FIGURES_DIR / "clean_accuracy.png"
    plot_clean_bar(data, out)
    print(f"Saved {out.name}")

    for model, rows in data.items():
        out = FIGURES_DIR / f"heatmap_{model}.png"
        plot_heatmap(rows, model, out)
        print(f"Saved {out.name}")

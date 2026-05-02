import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODELS = ["baseline_cnn", "resnet", "wider_cnn"]
STRATEGIES = ["baseline", "stdaug", "degaug"]

CSV_FILES = {
    ("baseline_cnn", "baseline"): RESULTS_DIR / "baseline_cnn_robustness_seed42.csv",
    ("resnet",        "baseline"): RESULTS_DIR / "resnet_robustness_seed123.csv",
    ("wider_cnn",     "baseline"): RESULTS_DIR / "wider_cnn_robustness_seed456.csv",
    ("baseline_cnn",  "stdaug"):   RESULTS_DIR / "baseline_cnn_stdaug_robustness.csv",
    ("resnet",        "stdaug"):   RESULTS_DIR / "resnet_stdaug_robustness.csv",
    ("wider_cnn",     "stdaug"):   RESULTS_DIR / "wider_cnn_stdaug_robustness.csv",
    ("baseline_cnn",  "degaug"):   RESULTS_DIR / "baseline_cnn_degaug_robustness.csv",
    ("resnet",        "degaug"):   RESULTS_DIR / "resnet_degaug_robustness.csv",
    ("wider_cnn",     "degaug"):   RESULTS_DIR / "wider_cnn_degaug_robustness.csv",
}

MODEL_LABELS = {
    "baseline_cnn": "BaselineCNN",
    "resnet":       "LightweightResNet",
    "wider_cnn":    "WiderCNN",
}

STRATEGY_LABELS = {
    "baseline": "Baseline",
    "stdaug":   "Std Aug",
    "degaug":   "Deg Aug",
}

STRATEGY_STYLES = {
    "baseline": {"color": "blue",   "linestyle": "-"},
    "stdaug":   {"color": "orange", "linestyle": "--"},
    "degaug":   {"color": "green",  "linestyle": ":"},
}

MODEL_COLORS = {
    "baseline_cnn": "steelblue",
    "resnet":       "darkorange",
    "wider_cnn":    "forestgreen",
}

CORRUPTION_TITLES = {
    "gaussian_noise":       "Gaussian Noise",
    "gaussian_blur":        "Gaussian Blur",
    "jpeg_compression":     "JPEG Compression",
    "resolution_reduction": "Resolution Reduction",
    "brightness":           "Brightness Change",
}

CORRUPTION_TYPES = list(CORRUPTION_TITLES.keys())
SEVERITIES = [1, 2, 3, 4, 5]


def load_csv(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["corruption_type"], int(row["severity"]))
            rows[key] = float(row["accuracy"])
    return rows


# ── Plot type 1: line plots (model × corruption type, 3 strategy lines each) ──

def plot_model_corruption_strategies(data, model, corruption_type, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy in STRATEGIES:
        rows = data[(model, strategy)]
        accuracies = [rows[(corruption_type, s)] for s in SEVERITIES]
        style = STRATEGY_STYLES[strategy]
        ax.plot(SEVERITIES, accuracies, marker="o",
                label=STRATEGY_LABELS[strategy], **style)
    ax.set_title(f"{MODEL_LABELS[model]} — {CORRUPTION_TITLES[corruption_type]}")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(SEVERITIES)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.grid(True)
    if corruption_type == "brightness":
        ax.axvline(x=3.5, color="gray", linestyle="--", linewidth=1)
        ax.text(2.25, 0.03, "darkening", ha="center", va="bottom", fontsize=8, color="gray")
        ax.text(4.75, 0.03, "brightening", ha="center", va="bottom", fontsize=8, color="gray")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Plot type 2: bar chart per corruption type (9 bars grouped by strategy) ──

def plot_corruption_strategy_comparison(data, corruption_type, out_path):
    n_models = len(MODELS)
    group_width = 0.7
    bar_width = group_width / n_models
    x = np.arange(len(STRATEGIES))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(MODELS):
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        heights = [
            np.mean([data[(model, strategy)][(corruption_type, s)] for s in SEVERITIES])
            for strategy in STRATEGIES
        ]
        ax.bar(offsets, heights, bar_width * 0.9,
               color=MODEL_COLORS[model], label=MODEL_LABELS[model])

    ax.set_title(f"{CORRUPTION_TITLES[corruption_type]} — Strategy Comparison")
    ax.set_xlabel("Training Strategy")
    ax.set_ylabel("Mean Accuracy (severities 1–5)")
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Plot type 3: overall 3×3 heatmap (model × strategy, mean over all corruptions) ──

def plot_overall_heatmap(data, out_path):
    matrix = np.array([
        [
            np.mean([
                data[(model, strategy)][(ct, s)]
                for ct in CORRUPTION_TYPES
                for s in SEVERITIES
            ])
            for strategy in STRATEGIES
        ]
        for model in MODELS
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_title("Mean Corruption Accuracy — Model × Strategy")
    ax.set_xlabel("Training Strategy")
    ax.set_ylabel("Model")
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])
    for i in range(len(MODELS)):
        for j in range(len(STRATEGIES)):
            ax.text(j, i, f"{matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=11)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Plot registry for --plots filtering ──

LINE_PLOT_KEYS = [f"{m}_{ct}" for m in MODELS for ct in CORRUPTION_TYPES]
BAR_PLOT_KEYS  = [f"{ct}_comparison" for ct in CORRUPTION_TYPES]
ALL_PLOTS = LINE_PLOT_KEYS + BAR_PLOT_KEYS + ["overall_heatmap"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plots", nargs="+", choices=ALL_PLOTS, default=None,
        metavar="PLOT",
        help=f"Which plots to generate. Default: all. Choices: {ALL_PLOTS}",
    )
    args = parser.parse_args()
    plots_to_run = set(args.plots) if args.plots else set(ALL_PLOTS)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = {key: load_csv(path) for key, path in CSV_FILES.items()}

    for model in MODELS:
        for corruption_type in CORRUPTION_TYPES:
            if f"{model}_{corruption_type}" not in plots_to_run:
                continue
            out = FIGURES_DIR / f"{model}_{corruption_type}_strategies.png"
            plot_model_corruption_strategies(data, model, corruption_type, out)
            print(f"Saved {out.name}")

    for corruption_type in CORRUPTION_TYPES:
        if f"{corruption_type}_comparison" not in plots_to_run:
            continue
        out = FIGURES_DIR / f"{corruption_type}_strategy_comparison.png"
        plot_corruption_strategy_comparison(data, corruption_type, out)
        print(f"Saved {out.name}")

    if "overall_heatmap" in plots_to_run:
        out = FIGURES_DIR / "overall_mean_accuracy_heatmap.png"
        plot_overall_heatmap(data, out)
        print(f"Saved {out.name}")

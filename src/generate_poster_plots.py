import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "figures" / "poster"
DPI = 180

COLOR_BASELINE = "#B0BEC5"
COLOR_STDAUG   = "#4A6FA5"
COLOR_DEGAUG   = "#B8940F"

STRATEGY_COLORS = [COLOR_BASELINE, COLOR_STDAUG, COLOR_DEGAUG]
STRATEGY_LABELS = ["Baseline", "Std Aug", "Deg Aug"]
STRATEGY_KEYS   = ["baseline", "stdaug", "degaug"]

MODELS      = ["baseline_cnn", "resnet", "wider_cnn"]
MODEL_LABELS = {"baseline_cnn": "BaselineCNN", "resnet": "LightweightResNet", "wider_cnn": "WiderCNN"}

CORRUPTION_TYPES = [
    "gaussian_noise",
    "gaussian_blur",
    "jpeg_compression",
    "resolution_reduction",
    "brightness",
]
CORRUPTION_LABELS = {
    "gaussian_noise":       "Gaussian Noise",
    "gaussian_blur":        "Gaussian Blur",
    "jpeg_compression":     "JPEG Compression",
    "resolution_reduction": "Resolution Reduction",
    "brightness":           "Brightness Change",
}
SEVERITIES = [1, 2, 3, 4, 5]

CSV_FILES = {
    ("baseline_cnn", "baseline"): RESULTS_DIR / "baseline_cnn_robustness_seed42.csv",
    ("resnet",        "baseline"): RESULTS_DIR / "resnet_robustness_seed42.csv",
    ("wider_cnn",     "baseline"): RESULTS_DIR / "wider_cnn_robustness_seed42.csv",
    ("baseline_cnn",  "stdaug"):   RESULTS_DIR / "baseline_cnn_stdaug_robustness.csv",
    ("resnet",        "stdaug"):   RESULTS_DIR / "resnet_stdaug_robustness.csv",
    ("wider_cnn",     "stdaug"):   RESULTS_DIR / "wider_cnn_stdaug_robustness.csv",
    ("baseline_cnn",  "degaug"):   RESULTS_DIR / "baseline_cnn_degaug_robustness.csv",
    ("resnet",        "degaug"):   RESULTS_DIR / "resnet_degaug_robustness.csv",
    ("wider_cnn",     "degaug"):   RESULTS_DIR / "wider_cnn_degaug_robustness.csv",
}


def load_csv(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[(row["corruption_type"], int(row["severity"]))] = float(row["accuracy"])
    return rows


def load_all_data():
    return {key: load_csv(path) for key, path in CSV_FILES.items()}


def corruption_means_sorted(data):
    """Mean accuracy per corruption type across all models/strategies/severities, sorted low→high."""
    means = {}
    for ct in CORRUPTION_TYPES:
        vals = [
            data[(m, s)][(ct, sev)]
            for m in MODELS
            for s in STRATEGY_KEYS
            for sev in SEVERITIES
        ]
        means[ct] = sum(vals) / len(vals)
    return sorted(means.items(), key=lambda x: x[1])


def model_strategy_means(data):
    """Mean accuracy per (strategy, model) across all corruptions/severities."""
    result = {}
    for strategy, label in zip(STRATEGY_KEYS, STRATEGY_LABELS):
        result[label] = [
            sum(data[(m, strategy)][(ct, sev)] for ct in CORRUPTION_TYPES for sev in SEVERITIES)
            / (len(CORRUPTION_TYPES) * len(SEVERITIES))
            for m in MODELS
        ]
    return result


def strategy_per_corruption_means(data):
    """Mean accuracy per (strategy, corruption_type) across all models/severities."""
    result = {}
    for strategy, label in zip(STRATEGY_KEYS, STRATEGY_LABELS):
        result[label] = [
            sum(data[(m, strategy)][(ct, sev)] for m in MODELS for sev in SEVERITIES)
            / (len(MODELS) * len(SEVERITIES))
            for ct in CORRUPTION_TYPES
        ]
    return result


def apply_style(ax):
    ax.set_facecolor("#FFFFFF")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#CCCCCC")
    ax.tick_params(colors="#444444")


def apply_h_grid(ax):
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.set_axisbelow(True)


# ── Plot 1: horizontal bar chart — mean accuracy by corruption type ────────────

def plot1(out_path, data):
    sorted_pairs = corruption_means_sorted(data)
    labels = [CORRUPTION_LABELS[ct] for ct, _ in sorted_pairs]
    values = [v for _, v in sorted_pairs]
    bar_colors = ["#B0BEC5", "#8EAFD4", "#4A6FA5", "#2E4E8A", "#1B2E4B"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#FFFFFF")

    bars = ax.barh(labels, values, color=bar_colors, height=0.55)
    for bar, v in zip(bars, values):
        ax.text(
            v + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{v*100:.1f}%",
            va="center", ha="left", fontsize=11, color="#333333",
        )

    ax.set_title("Mean Accuracy by Corruption Type", fontsize=14, pad=12)
    ax.set_xlabel("Mean Accuracy", fontsize=12)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.set_axisbelow(True)
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, facecolor="#FFFFFF")
    plt.close(fig)


# ── Plot 2: grouped bar chart — mean robustness by model and strategy ──────────

def plot2(out_path, data):
    model_labels = [MODEL_LABELS[m] for m in MODELS]
    strategy_data = model_strategy_means(data)

    n_models = len(MODELS)
    n_strategies = len(STRATEGY_LABELS)
    group_width = 0.65
    bar_width = group_width / n_strategies
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FFFFFF")

    for i, (label, color) in enumerate(zip(STRATEGY_LABELS, STRATEGY_COLORS)):
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        heights = strategy_data[label]
        bars = ax.bar(offsets, heights, bar_width * 0.9, color=color, label=label)
        for bar, h in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.006,
                f"{h*100:.1f}%",
                ha="center", va="bottom", fontsize=9, color="#333333",
            )

    ax.set_title("Mean Corruption Robustness by Model and Training Strategy",
                 fontsize=14, pad=12)
    ax.set_ylabel("Mean Accuracy", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    apply_h_grid(ax)
    apply_style(ax)
    ax.legend(fontsize=11, framealpha=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, facecolor="#FFFFFF")
    plt.close(fig)


# ── Plot 3: grouped bar chart — strategy per corruption type ───────────────────

def plot3(out_path, data):
    corruption_label_list = [CORRUPTION_LABELS[ct] for ct in CORRUPTION_TYPES]
    strategy_data = strategy_per_corruption_means(data)

    n_corruptions = len(CORRUPTION_TYPES)
    n_strategies = len(STRATEGY_LABELS)
    group_width = 0.65
    bar_width = group_width / n_strategies
    x = np.arange(n_corruptions)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#FFFFFF")

    for i, (label, color) in enumerate(zip(STRATEGY_LABELS, STRATEGY_COLORS)):
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        heights = strategy_data[label]
        bars = ax.bar(offsets, heights, bar_width * 0.9, color=color, label=label)
        if label == "Deg Aug":
            for bar, h in zip(bars, heights):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.006,
                    f"{h*100:.1f}%",
                    ha="center", va="bottom", fontsize=9, color="#333333",
                )

    ax.set_title("Effect of Training Strategy per Corruption Type",
                 fontsize=14, pad=12)
    ax.set_ylabel("Mean Accuracy", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(corruption_label_list, fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    apply_h_grid(ax)
    apply_style(ax)
    ax.legend(fontsize=11, framealpha=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, facecolor="#FFFFFF")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_all_data()

    plots = [
        ("plot1_corruption_overview.png",     plot1),
        ("plot2_model_strategy.png",          plot2),
        ("plot3_strategy_per_corruption.png", plot3),
    ]

    for filename, fn in plots:
        out = OUT_DIR / filename
        fn(out, data)
        print(f"Saved {out}")

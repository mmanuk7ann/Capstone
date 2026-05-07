from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures" / "poster"
DPI = 180

COLOR_BASELINE = "#B0BEC5"
COLOR_STDAUG   = "#4A6FA5"
COLOR_DEGAUG   = "#B8940F"

STRATEGY_COLORS = [COLOR_BASELINE, COLOR_STDAUG, COLOR_DEGAUG]
STRATEGY_LABELS = ["Baseline", "Std Aug", "Deg Aug"]


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

def plot1(out_path):
    labels = [
        "Gaussian Noise",
        "Resolution Reduction",
        "Gaussian Blur",
        "JPEG Compression",
        "Brightness Change",
    ]
    values = [0.398, 0.552, 0.561, 0.668, 0.678]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#FFFFFF")

    bar_colors = ["#B0BEC5", "#8EAFD4", "#4A6FA5", "#2E4E8A", "#1B2E4B"]
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

def plot2(out_path):
    models = ["BaselineCNN", "LightweightResNet", "WiderCNN"]
    data = {
        "Baseline": [0.534, 0.448, 0.523],
        "Std Aug":  [0.533, 0.449, 0.512],
        "Deg Aug":  [0.707, 0.660, 0.775],
    }

    n_models = len(models)
    n_strategies = len(STRATEGY_LABELS)
    group_width = 0.65
    bar_width = group_width / n_strategies
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FFFFFF")

    for i, (label, color) in enumerate(zip(STRATEGY_LABELS, STRATEGY_COLORS)):
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        heights = data[label]
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
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    apply_h_grid(ax)
    apply_style(ax)
    ax.legend(fontsize=11, framealpha=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, facecolor="#FFFFFF")
    plt.close(fig)


# ── Plot 3: grouped bar chart — strategy per corruption type ───────────────────

def plot3(out_path):
    corruption_labels = [
        "Gaussian Noise",
        "Gaussian Blur",
        "JPEG Compression",
        "Resolution Reduction",
        "Brightness Change",
    ]
    data = {
        "Baseline": [0.3033, 0.4780, 0.6382, 0.4549, 0.6343],
        "Std Aug":  [0.2619, 0.4798, 0.6418, 0.4651, 0.6398],
        "Deg Aug":  [0.6276, 0.7243, 0.7247, 0.7346, 0.7592],
    }

    n_corruptions = len(corruption_labels)
    n_strategies = len(STRATEGY_LABELS)
    group_width = 0.65
    bar_width = group_width / n_strategies
    x = np.arange(n_corruptions)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#FFFFFF")

    for i, (label, color) in enumerate(zip(STRATEGY_LABELS, STRATEGY_COLORS)):
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        heights = data[label]
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
    ax.set_xticklabels(corruption_labels, fontsize=11)
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

    plots = [
        ("plot1_corruption_overview.png",   plot1),
        ("plot2_model_strategy.png",        plot2),
        ("plot3_strategy_per_corruption.png", plot3),
    ]

    for filename, fn in plots:
        out = OUT_DIR / filename
        fn(out)
        print(f"Saved {out}")

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_DIR   = PROJECT_ROOT / "models"

TRAIN_SCRIPTS = [
    "train_baseline_cnn.py",
    "train_resnet.py",
    "train_wider_cnn.py",
    "train_baseline_cnn_stdaug.py",
    "train_resnet_stdaug.py",
    "train_wider_cnn_stdaug.py",
    "train_baseline_cnn_degaug.py",
    "train_resnet_degaug.py",
    "train_wider_cnn_degaug.py",
]

# (architecture_name, weights_filename, output_csv_name)
EVALUATE_CONFIGS = [
    ("baseline_cnn", "baseline_cnn_seed42_best.pth",       "baseline_cnn_robustness_seed42.csv"),
    ("resnet",       "resnet_seed123_best.pth",             "resnet_robustness_seed42.csv"),
    ("wider_cnn",    "wider_cnn_seed456_best.pth",          "wider_cnn_robustness_seed42.csv"),
    ("baseline_cnn", "baseline_cnn_stdaug_seed42_best.pth", "baseline_cnn_stdaug_robustness.csv"),
    ("resnet",       "resnet_stdaug_seed42_best.pth",       "resnet_stdaug_robustness.csv"),
    ("wider_cnn",    "wider_cnn_stdaug_seed42_best.pth",    "wider_cnn_stdaug_robustness.csv"),
    ("baseline_cnn", "baseline_cnn_degaug_seed42_best.pth", "baseline_cnn_degaug_robustness.csv"),
    ("resnet",       "resnet_degaug_seed42_best.pth",       "resnet_degaug_robustness.csv"),
    ("wider_cnn",    "wider_cnn_degaug_seed42_best.pth",    "wider_cnn_degaug_robustness.csv"),
]


def run(cmd, desc):
    print(f"\n{'='*64}")
    print(f"  {desc}")
    print(f"{'='*64}")
    t0 = time.time()
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    print(f"  Done in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full CIFAR-10 robustness evaluation pipeline."
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training and use existing .pth weight files.",
    )
    args = parser.parse_args()

    for d in [
        MODELS_DIR,
        RESULTS_DIR,
        RESULTS_DIR / "figures",
        RESULTS_DIR / "figures" / "poster",
    ]:
        d.mkdir(parents=True, exist_ok=True)
    print("Directories ready.")

    py = sys.executable
    pipeline_start = time.time()

    # ── Training ──────────────────────────────────────────────────────────────
    if args.skip_training:
        print("\nSkipping training (--skip-training flag set).")
    else:
        for script in TRAIN_SCRIPTS:
            run(
                [py, f"src/{script}", "--seed", "42"],
                f"Training: {script}",
            )

    # ── Evaluation ────────────────────────────────────────────────────────────
    for arch, weights_file, output_csv in EVALUATE_CONFIGS:
        weights_path = MODELS_DIR / weights_file
        output_path  = RESULTS_DIR / output_csv
        run(
            [
                py, "src/evaluate_robustness.py",
                "--model",   arch,
                "--weights", str(weights_path),
                "--output",  str(output_path),
            ],
            f"Evaluating: {output_csv}",
        )

    # ── Visualisation ─────────────────────────────────────────────────────────
    run([py, "src/visualize_robustness.py"], "Generating robustness figures")
    run([py, "src/generate_poster_plots.py"], "Generating poster figures")
    run([py, "src/compute_poster_stats.py"], "Computing poster statistics")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    print(f"\n{'='*64}")
    print(f"  Pipeline complete in {total / 60:.1f} min")
    print(f"  Models     → {MODELS_DIR}")
    print(f"  Results    → {RESULTS_DIR}")
    print(f"  Figures    → {RESULTS_DIR / 'figures'}")
    print(f"  Poster     → {RESULTS_DIR / 'figures' / 'poster'}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()

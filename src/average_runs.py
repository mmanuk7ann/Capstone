import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODELS = ["baseline_cnn", "resnet", "wider_cnn"]
SEEDS  = [42, 123, 456]


def load_metrics(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch":      int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "test_loss":  float(row["test_loss"]),
                "accuracy":   float(row["accuracy"]),
            })
    return rows


def average_seeds(model):
    """Load metrics for all available seeds and return per-epoch mean/std of accuracy."""
    all_runs = []
    found_seeds = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"{model}_seed{seed}_metrics.csv"
        if path.exists():
            all_runs.append(load_metrics(path))
            found_seeds.append(seed)

    if not all_runs:
        return None, []

    n_epochs = len(all_runs[0])
    results = []
    for i in range(n_epochs):
        epoch = all_runs[0][i]["epoch"]
        accs = [run[i]["accuracy"] for run in all_runs]
        mean = sum(accs) / len(accs)
        if len(accs) > 1:
            variance = sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
            std = variance ** 0.5
        else:
            std = 0.0
        results.append({"epoch": epoch, "mean_accuracy": mean, "std_accuracy": std})

    return results, found_seeds


def print_table(model, results, seeds):
    col_w = 16
    label_w = 8
    header = f"{'Epoch':<{label_w}}{'Mean Acc':>{col_w}}{'Std Acc':>{col_w}}"
    separator = "-" * len(header)
    print(f"\n{model.upper()}  (seeds: {seeds})")
    print(separator)
    print(header)
    print(separator)
    step = max(1, len(results) // 10)
    for row in results[::step]:
        print(
            f"{row['epoch']:<{label_w}}"
            f"{row['mean_accuracy']:>{col_w}.4f}"
            f"{row['std_accuracy']:>{col_w}.4f}"
        )
    last = results[-1]
    if last["epoch"] != results[((len(results) - 1) // step) * step]["epoch"]:
        print(
            f"{last['epoch']:<{label_w}}"
            f"{last['mean_accuracy']:>{col_w}.4f}"
            f"{last['std_accuracy']:>{col_w}.4f}"
        )
    print(separator)
    print(f"  Final mean accuracy: {last['mean_accuracy']:.4f} ± {last['std_accuracy']:.4f}")


def save_csv(model, results, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mean_accuracy", "std_accuracy"])
        for row in results:
            writer.writerow([row["epoch"], f"{row['mean_accuracy']:.6f}", f"{row['std_accuracy']:.6f}"])
    print(f"  Saved → {path}")


if __name__ == "__main__":
    for model in MODELS:
        results, seeds = average_seeds(model)
        if results is None:
            print(f"\n{model.upper()}: no metric CSVs found — skipping.")
            continue
        print_table(model, results, seeds)
        out_path = RESULTS_DIR / f"{model}_averaged_metrics.csv"
        save_csv(model, results, out_path)

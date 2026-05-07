import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

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

STRATEGY_FILES = {
    "Baseline": [
        RESULTS_DIR / "baseline_cnn_robustness_seed42.csv",
        RESULTS_DIR / "resnet_robustness_seed123.csv",
        RESULTS_DIR / "wider_cnn_robustness_seed456.csv",
    ],
    "StdAug": [
        RESULTS_DIR / "baseline_cnn_stdaug_robustness.csv",
        RESULTS_DIR / "resnet_stdaug_robustness.csv",
        RESULTS_DIR / "wider_cnn_stdaug_robustness.csv",
    ],
    "DegAug": [
        RESULTS_DIR / "baseline_cnn_degaug_robustness.csv",
        RESULTS_DIR / "resnet_degaug_robustness.csv",
        RESULTS_DIR / "wider_cnn_degaug_robustness.csv",
    ],
}

WIDER_CNN_BASELINE_FILE = RESULTS_DIR / "wider_cnn_robustness_seed456.csv"

STRATEGIES = ["Baseline", "StdAug", "DegAug"]


def load_csv(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[(row["corruption_type"], int(row["severity"]))] = float(row["accuracy"])
    return rows


def compute_table1():
    table = {}
    for strategy, files in STRATEGY_FILES.items():
        datasets = [load_csv(p) for p in files]
        for ct in CORRUPTION_TYPES:
            vals = [
                d[(ct, s)]
                for d in datasets
                for s in SEVERITIES
            ]
            table[(ct, strategy)] = sum(vals) / len(vals)
    return table


def compute_table2():
    rows = load_csv(WIDER_CNN_BASELINE_FILE)
    table = {}
    for ct in CORRUPTION_TYPES:
        for s in SEVERITIES:
            table[(ct, s)] = rows[(ct, s)]
    return table


def print_table1(table):
    col_w = 12
    label_w = 22
    header = f"{'Corruption Type':<{label_w}}" + "".join(f"{s:>{col_w}}" for s in STRATEGIES)
    separator = "-" * len(header)
    print("\nTable 1 — Mean Accuracy per Corruption Type per Strategy (avg over 3 models)")
    print(separator)
    print(header)
    print(separator)
    for ct in CORRUPTION_TYPES:
        row = f"{CORRUPTION_LABELS[ct]:<{label_w}}"
        for s in STRATEGIES:
            row += f"{table[(ct, s)]:>{col_w}.4f}"
        print(row)
    print(separator)


def print_table2(table):
    col_w = 10
    label_w = 22
    header = f"{'Corruption Type':<{label_w}}" + "".join(f"{'Sev ' + str(s):>{col_w}}" for s in SEVERITIES)
    separator = "-" * len(header)
    print("\nTable 2 — WiderCNN Baseline Accuracy per Corruption × Severity")
    print(separator)
    print(header)
    print(separator)
    for ct in CORRUPTION_TYPES:
        row = f"{CORRUPTION_LABELS[ct]:<{label_w}}"
        for s in SEVERITIES:
            row += f"{table[(ct, s)]:>{col_w}.4f}"
        print(row)
    print(separator)


def save_table1(table, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["corruption_type"] + STRATEGIES)
        for ct in CORRUPTION_TYPES:
            writer.writerow([ct] + [f"{table[(ct, s)]:.6f}" for s in STRATEGIES])
    print(f"\nSaved Table 1 → {path}")


def save_table2(table, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["corruption_type"] + [f"severity_{s}" for s in SEVERITIES])
        for ct in CORRUPTION_TYPES:
            writer.writerow([ct] + [f"{table[(ct, s)]:.6f}" for s in SEVERITIES])
    print(f"Saved Table 2 → {path}")


if __name__ == "__main__":
    table1 = compute_table1()
    table2 = compute_table2()

    print_table1(table1)
    print_table2(table2)

    save_table1(table1, RESULTS_DIR / "poster_strategy_comparison.csv")
    save_table2(table2, RESULTS_DIR / "wider_cnn_baseline_heatmap_data.csv")

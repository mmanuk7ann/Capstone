import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data_loader import NORMALIZE_MEAN, NORMALIZE_STD, BATCH_SIZE
from corruption_transforms import CORRUPTION_TYPES, SEVERITY_LEVELS, get_corruption
from models.baseline_cnn import BaselineCNN
from models.resnet_cnn import LightweightResNet
from models.wider_cnn import WiderCNN

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MODEL_MAP = {
    "baseline_cnn": BaselineCNN,
    "resnet": LightweightResNet,
    "wider_cnn": WiderCNN,
}


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


def make_corrupted_loader(corruption_type, severity):
    transform = transforms.Compose([
        get_corruption(corruption_type, severity),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    dataset = datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform, download=True)
    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_MAP.keys())
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    model = MODEL_MAP[args.model]().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from {args.weights}")

    args.output.parent.mkdir(exist_ok=True)
    results_file = args.output

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("corruption_type,severity,accuracy\n")

        clean_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
        clean_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, transform=clean_transform, download=True)
        clean_loader = DataLoader(clean_dataset, batch_size=BATCH_SIZE, num_workers=2)
        clean_acc = evaluate(model, clean_loader, device)
        f.write(f"clean,0,{clean_acc}\n")
        print(f"clean | severity 0 | accuracy: {clean_acc:.4f}")

        for corruption_type in CORRUPTION_TYPES:
            for severity in SEVERITY_LEVELS:
                loader = make_corrupted_loader(corruption_type, severity)
                acc = evaluate(model, loader, device)
                f.write(f"{corruption_type},{severity},{acc}\n")
                print(f"{corruption_type} | severity {severity} | accuracy: {acc:.4f}")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()

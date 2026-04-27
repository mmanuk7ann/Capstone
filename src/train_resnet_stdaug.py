import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data_loader import (
    test_loader,
    train_dataset, test_dataset, classes,
    BATCH_SIZE, LEARNING_RATE, NORMALIZE_MEAN, NORMALIZE_STD,
)
from models.resnet_cnn import LightweightResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 40
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    train_loader = DataLoader(
        datasets.CIFAR10(root=DATA_DIR, train=True, transform=transform_train, download=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    )

    RESULTS_FILE = RESULTS_DIR / f"resnet_stdaug_seed{args.seed}_metrics.csv"
    BEST_MODEL_FILE = MODELS_DIR / f"resnet_stdaug_seed{args.seed}_best.pth"
    FINAL_MODEL_FILE = MODELS_DIR / f"resnet_stdaug_seed{args.seed}_final.pth"
    CONFIG_FILE = RESULTS_DIR / f"resnet_stdaug_seed{args.seed}_config.json"

    print("Train Samples: ", len(train_dataset))
    print("Tests Samples: ", len(test_dataset))
    print("Classes: ", classes)

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        file.write("epoch,train_loss,test_loss,accuracy\n")

    config = {
        "model_name": "LightweightResNet",
        "dataset": "CIFAR-10",
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss_function": "CrossEntropyLoss",
        "normalization_mean": list(NORMALIZE_MEAN),
        "normalization_std": list(NORMALIZE_STD),
        "scheduler": "CosineAnnealingLR",
        "seed": args.seed,
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"Training on: {device}")

    resnet_model = LightweightResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy = 0.0

    for epoch in range(epochs):
        resnet_model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        resnet_model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = resnet_model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        with open(RESULTS_FILE, "a", encoding="utf-8") as file:
            file.write(f"{epoch + 1},{avg_train_loss},{avg_test_loss},{accuracy}\n")

        scheduler.step()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(resnet_model.state_dict(), BEST_MODEL_FILE)

        print(
            f"Epoch: {epoch + 1}/{epochs}, "
            f"Epoch Train loss: {avg_train_loss}, "
            f"Epoch Test loss: {avg_test_loss}, "
            f"Epoch Accuracy: {accuracy}"
        )

    torch.save(resnet_model.state_dict(), FINAL_MODEL_FILE)

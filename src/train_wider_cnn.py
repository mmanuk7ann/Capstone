import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

from data_loader import (
    train_loader, test_loader,
    train_dataset, test_dataset, classes,
    BATCH_SIZE, LEARNING_RATE, NORMALIZE_MEAN, NORMALIZE_STD,
)
from models.wider_cnn import WiderCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 40
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    RESULTS_FILE = RESULTS_DIR / f"wider_cnn_seed{args.seed}_metrics.csv"
    BEST_MODEL_FILE = MODELS_DIR / f"wider_cnn_seed{args.seed}_best.pth"
    FINAL_MODEL_FILE = MODELS_DIR / f"wider_cnn_seed{args.seed}_final.pth"
    CONFIG_FILE = RESULTS_DIR / f"wider_cnn_seed{args.seed}_config.json"

    print("Train Samples: ", len(train_dataset))
    print("Tests Samples: ", len(test_dataset))
    print("Classes: ", classes)

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        file.write("epoch,train_loss,test_loss,accuracy\n")

    config = {
        "model_name": "WiderCNN",
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

    wider_cnn_model = WiderCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(wider_cnn_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_accuracy = 0.0

    for epoch in range(epochs):
        wider_cnn_model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = wider_cnn_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        wider_cnn_model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = wider_cnn_model(images)
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
            torch.save(wider_cnn_model.state_dict(), BEST_MODEL_FILE)

        print(
            f"Epoch: {epoch + 1}/{epochs}, "
            f"Epoch Train loss: {avg_train_loss}, "
            f"Epoch Test loss: {avg_test_loss}, "
            f"Epoch Accuracy: {accuracy}"
        )

    torch.save(wider_cnn_model.state_dict(), FINAL_MODEL_FILE)

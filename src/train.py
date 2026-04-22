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
from models.baseline_cnn import BaselineCNN


epochs = 30
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_FILE = RESULTS_DIR / "baseline_cnn_metrics.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BEST_MODEL_FILE = MODELS_DIR / "baseline_cnn_best.pth"
FINAL_MODEL_FILE = MODELS_DIR / "baseline_cnn_final.pth"
CONFIG_FILE = RESULTS_DIR / "baseline_cnn_config.json"

if __name__ == "__main__":
    print("Train Samples: ", len(train_dataset))
    print("Tests Samples: ", len(test_dataset))
    print("Classes: ", classes)

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        file.write("epoch,train_loss,test_loss,accuracy\n")

    config = {
        "model_name": "BaselineCNN",
        "dataset": "CIFAR-10",
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss_function": "CrossEntropyLoss",
        "normalization_mean": list(NORMALIZE_MEAN),
        "normalization_std": list(NORMALIZE_STD),
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    baseline_model = BaselineCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)
    best_accuracy = 0.0

    for epoch in range(epochs):
        baseline_model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = baseline_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        baseline_model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = baseline_model(images)
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

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(baseline_model.state_dict(), BEST_MODEL_FILE)

        print(
            f"Epoch: {epoch + 1}/{epochs}, "
            f"Epoch Train loss: {avg_train_loss}, "
            f"Epoch Test loss: {avg_test_loss}, "
            f"Epoch Accuracy: {accuracy}"
        )

    torch.save(baseline_model.state_dict(), FINAL_MODEL_FILE)

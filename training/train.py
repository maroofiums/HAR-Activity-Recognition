"""
train.py

Training loop for the CNN-LSTM Human Activity Recognition model.

Usage:
    python training/train.py
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import mlflow
import mlflow.pytorch

# Allow importing from project root when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data_loader import build_datasets, NUM_CLASSES
from models.cnn_lstm import CNNLSTM


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def subject_based_split(raw_root: str, val_fraction: float = 0.15, seed: int = 42):
    """
    Splits training subjects into train/val groups, so that all windows
    from a given subject end up entirely in train OR entirely in val.

    Returns:
        train_indices, val_indices (both as np.ndarray of row indices
        into the full training set)
    """
    subject_path = os.path.join(raw_root, "train", "subject_train.txt")
    subjects = np.loadtxt(subject_path).astype(int)  # shape (N,)

    unique_subjects = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)

    num_val_subjects = max(1, int(len(unique_subjects) * val_fraction))
    val_subjects = set(unique_subjects[:num_val_subjects])

    train_indices = np.where(~np.isin(subjects, list(val_subjects)))[0]
    val_indices = np.where(np.isin(subjects, list(val_subjects)))[0]

    return train_indices, val_indices


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights, normalized to mean 1.0.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)  # avoid div by zero
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, dataloader, criterion, device) -> tuple[float, float]:
    """
    Runs evaluation over a dataloader. Returns (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += x_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_root = config["raw_root"]

    # --- Build full train/test datasets (normalization computed on full train) ---
    full_train_dataset, test_dataset, stats = build_datasets(raw_root)

    # --- Subject-based train/val split ---
    train_idx, val_idx = subject_based_split(
        raw_root, val_fraction=config["val_fraction"], seed=config["seed"]
    )
    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)

    print(f"Train windows: {len(train_subset)}")
    print(f"Val windows:   {len(val_subset)}")
    print(f"Test windows:  {len(test_dataset)}")

    train_loader = DataLoader(
        train_subset, batch_size=config["batch_size"], shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_subset, batch_size=config["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # --- Model, loss, optimizer ---
    model = CNNLSTM(
        in_channels=9,
        num_classes=NUM_CLASSES,
        lstm_hidden_size=config["lstm_hidden_size"],
        lstm_num_layers=config["lstm_num_layers"],
        dropout=config["dropout"],
    ).to(device)

    # Class weights computed from the actual training subset labels
    train_labels = full_train_dataset.y[train_idx].numpy()
    class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # --- Early stopping state ---
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "cnn_lstm_best.pt")

    mlflow.set_experiment("har_cnn_lstm")

    with mlflow.start_run():
        mlflow.log_params({
            "lstm_hidden_size": config["lstm_hidden_size"],
            "lstm_num_layers": config["lstm_num_layers"],
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "batch_size": config["batch_size"],
        })

        for epoch in range(1, config["max_epochs"] + 1):
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x_batch.size(0)
                preds = logits.argmax(dim=1)
                running_correct += (preds == y_batch).sum().item()
                running_total += x_batch.size(0)

            train_loss = running_loss / running_total
            train_acc = running_correct / running_total

            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            # --- Early stopping check ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config["early_stopping_patience"]:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        # --- Final test evaluation using best checkpoint ---
        model.load_state_dict(torch.load(best_checkpoint_path))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"\nFinal Test | test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    train(config)

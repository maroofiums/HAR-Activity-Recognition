"""
data_loader.py

Loads the UCI HAR raw Inertial Signals dataset and prepares it as a
PyTorch Dataset for the CNN-LSTM activity recognition model.

Directory expected (relative to project root):
    data/raw/UCI HAR Dataset/train/Inertial Signals/*.txt
    data/raw/UCI HAR Dataset/train/y_train.txt
    data/raw/UCI HAR Dataset/test/Inertial Signals/*.txt
    data/raw/UCI HAR Dataset/test/y_test.txt
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Fixed ordering of the 9 raw signal channels.
# This order defines what channel index 0..8 means throughout the project.
SIGNAL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

NUM_CLASSES = 6

ACTIVITY_LABELS = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}


def _load_signal_file(filepath: str) -> np.ndarray:
    """
    Loads a single Inertial Signals file.
    Each line has 128 space-separated float values (one window).
    Returns array of shape (N, 128).
    """
    return np.loadtxt(filepath)


def load_split(raw_root: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads one split ("train" or "test") of the UCI HAR dataset.

    Returns:
        X: np.ndarray of shape (N, 128, 9)
        y: np.ndarray of shape (N,), zero-indexed in [0, 5]
    """
    assert split in ("train", "test")

    signals_dir = os.path.join(raw_root, split, "Inertial Signals")
    labels_path = os.path.join(raw_root, split, f"y_{split}.txt")

    channel_arrays = []
    for signal_name in SIGNAL_NAMES:
        file_path = os.path.join(signals_dir, f"{signal_name}_{split}.txt")
        arr = _load_signal_file(file_path)  # shape (N, 128)
        channel_arrays.append(arr)

    # Stack along a new last axis -> (N, 128, 9)
    X = np.stack(channel_arrays, axis=-1).astype(np.float32)

    y = np.loadtxt(labels_path).astype(np.int64)
    y = y - 1  # convert 1..6 labels to 0..5 for CrossEntropyLoss

    return X, y


def compute_normalization_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes per-channel mean and std from the training set only.

    X_train shape: (N, 128, 9)
    Returns:
        mean: shape (9,)
        std:  shape (9,)
    """
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))
    std = np.where(std < 1e-8, 1e-8, std)  # guard against divide-by-zero
    return mean, std


def apply_normalization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Applies z-score normalization per channel: (x - mean) / std
    """
    return (X - mean) / std


class HARDataset(Dataset):
    """
    PyTorch Dataset wrapping normalized HAR windows.

    __getitem__ returns:
        x: FloatTensor of shape (128, 9)   [time, channels]
        y: LongTensor scalar               [class index 0..5]
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_datasets(raw_root: str) -> tuple[HARDataset, HARDataset, dict]:
    """
    Full pipeline: load raw train/test, compute stats from train,
    normalize both splits, and return ready-to-use Dataset objects.

    Returns:
        train_dataset, test_dataset, stats
        stats = {"mean": np.ndarray(9,), "std": np.ndarray(9,)}
    """
    X_train, y_train = load_split(raw_root, "train")
    X_test, y_test = load_split(raw_root, "test")

    mean, std = compute_normalization_stats(X_train)

    X_train_norm = apply_normalization(X_train, mean, std)
    X_test_norm = apply_normalization(X_test, mean, std)

    train_dataset = HARDataset(X_train_norm, y_train)
    test_dataset = HARDataset(X_test_norm, y_test)

    stats = {"mean": mean, "std": std}
    return train_dataset, test_dataset, stats


if __name__ == "__main__":
    RAW_ROOT = os.path.join("data", "raw", "UCI HAR Dataset")

    train_ds, test_ds, stats = build_datasets(RAW_ROOT)

    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    x_sample, y_sample = train_ds[0]
    print(f"Sample X shape: {x_sample.shape}")
    print(f"Sample y value: {y_sample.item()}")

    print(f"Per-channel mean: {stats['mean']}")
    print(f"Per-channel std:  {stats['std']}")

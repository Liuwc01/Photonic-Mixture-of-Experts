from pathlib import Path

import numpy as np
import scipy.io as io
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

try:
    from .config import (
        DEFAULT_DATA_DIR,
        DEFAULT_SEED,
        DEFAULT_TEST_SIZE,
        FEATURE_MAP_PIXELS,
        FEATURE_MAP_SIZE,
        TASKS,
        WEIGHTED_CHANNELS,
    )
except ImportError:
    from config import (
        DEFAULT_DATA_DIR,
        DEFAULT_SEED,
        DEFAULT_TEST_SIZE,
        FEATURE_MAP_PIXELS,
        FEATURE_MAP_SIZE,
        TASKS,
        WEIGHTED_CHANNELS,
    )


def _labels_to_samples(raw_labels, num_samples):
    labels = np.asarray(raw_labels).reshape(-1)
    if labels.size == num_samples:
        return labels.astype(np.int64)
    if labels.size >= num_samples * FEATURE_MAP_PIXELS:
        return labels[: num_samples * FEATURE_MAP_PIXELS : FEATURE_MAP_PIXELS].astype(np.int64)
    if labels.size > num_samples:
        return labels[:num_samples].astype(np.int64)
    raise ValueError(f"Cannot derive {num_samples} sample labels from label array of length {labels.size}.")


def _load_single_pen(data_dir, task, num_channels):
    path = Path(data_dir) / f"{task.key}_output_{num_channels}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    mat = io.loadmat(path)
    if "train_x" not in mat or "labels" not in mat:
        raise KeyError(f"{path} must contain 'train_x' and 'labels'. Found keys: {list(mat.keys())}")

    raw_x = mat["train_x"].astype(np.float32)
    expected_rows = task.num_samples * FEATURE_MAP_PIXELS
    if raw_x.shape[0] < expected_rows or raw_x.shape[1] != num_channels:
        raise ValueError(
            f"{path} has shape {raw_x.shape}; expected at least ({expected_rows}, {num_channels})."
        )

    x = raw_x[:expected_rows].reshape(task.num_samples, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, num_channels)
    x = x.transpose(0, 3, 1, 2)
    y = _labels_to_samples(mat["labels"], task.num_samples)
    return x, y


def _split_tensor_dataset(x, y, batch_size, seed, test_size):
    train_x, test_x, train_y, test_y = train_test_split(
        x,
        y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=seed,
    )
    train_ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
    train_eval_ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        "train_eval": DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False),
    }


def build_hard_routing_loaders(data_dir=DEFAULT_DATA_DIR, seed=DEFAULT_SEED, test_size=DEFAULT_TEST_SIZE):
    loaders = {}
    for task in TASKS:
        x, y = _load_single_pen(data_dir, task, task.hard_channels)
        loaders[task.key] = _split_tensor_dataset(x, y, task.batch_size, seed, test_size)
    return loaders


def build_weighted_routing_loaders(data_dir=DEFAULT_DATA_DIR, seed=DEFAULT_SEED, test_size=DEFAULT_TEST_SIZE):
    loaders = {}
    for task in TASKS:
        feature_blocks = []
        labels = None
        for channels in WEIGHTED_CHANNELS:
            x, y = _load_single_pen(data_dir, task, channels)
            feature_blocks.append(x)
            if labels is None:
                labels = y
            elif not np.array_equal(labels, y):
                raise ValueError(f"Label mismatch across PEN files for task '{task.key}'.")
        x18 = np.concatenate(feature_blocks, axis=1)
        loaders[task.key] = _split_tensor_dataset(x18, labels, task.batch_size, seed, test_size)
    return loaders

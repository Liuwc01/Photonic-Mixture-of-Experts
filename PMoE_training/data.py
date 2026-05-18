from pathlib import Path

import numpy as np
import scipy.io as io
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

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
    data_dir = Path(data_dir)
    path = data_dir / f"{task.key}_output_{num_channels}.mat"
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


def _dbm_to_mw(values):
    return 10 ** (values / 10)


def _normalize_features(features):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1, std)
    return (features - mean) / std


class WeightedPhotonicDataset(Dataset):
    def __init__(self, features, labels, num_channels=None):
        self.features = features
        self.labels = labels
        self.num_channels = num_channels
        self.channel_idx = {4: 0, 6: 1, 8: 2}.get(num_channels, 0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        if self.num_channels is None:
            unified = features
        elif self.num_channels == 4:
            unified = torch.zeros(18, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
            unified[0:4] = features
        elif self.num_channels == 6:
            unified = torch.zeros(18, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
            unified[4:10] = features
        elif self.num_channels == 8:
            unified = torch.zeros(18, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
            unified[10:18] = features
        else:
            raise ValueError(f"Unsupported weighted-routing channel count: {self.num_channels}")
        return unified, torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(self.channel_idx, dtype=torch.long)


def _load_weighted_pen_dense(data_dir, task, num_channels):
    data_dir = Path(data_dir)
    path = data_dir / f"{task.key}_output_{num_channels}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    mat = io.loadmat(path)
    raw_x = mat["train_x"].astype(np.float32)
    expected_rows = task.num_samples * FEATURE_MAP_PIXELS
    if raw_x.shape[0] < expected_rows or raw_x.shape[1] != num_channels:
        raise ValueError(
            f"{path} has shape {raw_x.shape}; expected at least ({expected_rows}, {num_channels})."
        )

    features = raw_x[:expected_rows].reshape(task.num_samples, FEATURE_MAP_PIXELS, num_channels)
    features = _normalize_features(_dbm_to_mw(features))
    features = features.reshape(task.num_samples, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, num_channels)
    features = features.transpose(0, 3, 1, 2)

    y = _labels_to_samples(mat["labels"], task.num_samples)
    return features, y


def _load_weighted_pen_fused(data_dir, task):
    fused = np.zeros((task.num_samples, 18, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE), dtype=np.float32)
    split_labels = None
    for channels in WEIGHTED_CHANNELS:
        x, y = _load_weighted_pen_dense(data_dir, task, channels)
        if split_labels is None:
            split_labels = y
        elif not np.array_equal(split_labels, y):
            raise ValueError(f"Label mismatch across weighted-routing PEN outputs for {task.key}.")
        if channels == 4:
            fused[:, 0:4] = x
        elif channels == 6:
            fused[:, 4:10] = x
        elif channels == 8:
            fused[:, 10:18] = x
        else:
            raise ValueError(f"Unsupported weighted-routing channel count: {channels}")
    return fused, split_labels


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


def _split_tensor_dataset_random(x, y, batch_size, seed, test_size, channel_idx=None):
    tensors = [torch.tensor(x), torch.tensor(y, dtype=torch.long)]
    if channel_idx is not None:
        tensors.append(torch.tensor(channel_idx, dtype=torch.long))
    dataset = TensorDataset(*tensors)
    train_len = int(len(dataset) * (1 - test_size))
    test_len = len(dataset) - train_len
    train_ds, test_ds = random_split(dataset, [train_len, test_len])
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        "train_eval": DataLoader(train_ds, batch_size=batch_size, shuffle=False),
    }


def build_hard_routing_loaders(data_dir=DEFAULT_DATA_DIR, seed=DEFAULT_SEED, test_size=DEFAULT_TEST_SIZE):
    loaders = {}
    for task in TASKS:
        x, y = _load_single_pen(data_dir, task, task.hard_channels)
        loaders[task.key] = _split_tensor_dataset(x, y, task.batch_size, seed, test_size)
    return loaders


def build_weighted_routing_loaders(data_dir=DEFAULT_DATA_DIR, seed=DEFAULT_SEED, test_size=DEFAULT_TEST_SIZE):
    loaders = {}
    task_by_key = {task.key: task for task in TASKS}
    for task in [task_by_key["mnist"], task_by_key["emnist"], task_by_key["fmnist"]]:
        x, y = _load_weighted_pen_fused(data_dir, task)
        train_x, test_x, train_y, test_y = train_test_split(
            x,
            y,
            test_size=test_size,
            shuffle=True,
            stratify=y,
            random_state=seed,
        )
        train_ds = WeightedPhotonicDataset(train_x, train_y)
        test_ds = WeightedPhotonicDataset(test_x, test_y)
        loaders[task.key] = {
            "train": DataLoader(train_ds, batch_size=task.batch_size, shuffle=True, drop_last=True),
            "test": DataLoader(test_ds, batch_size=task.batch_size, shuffle=False),
            "train_eval": DataLoader(train_ds, batch_size=task.batch_size, shuffle=False),
        }
    return loaders

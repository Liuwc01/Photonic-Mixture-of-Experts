import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

try:
    from .config import DEFAULT_DATA_DIR, FEATURE_MAP_PIXELS, FEATURE_MAP_SIZE, TASKS, WEIGHTED_CHANNELS
    from .data import _dbm_to_mw
except ImportError:
    from config import DEFAULT_DATA_DIR, FEATURE_MAP_PIXELS, FEATURE_MAP_SIZE, TASKS, WEIGHTED_CHANNELS
    from data import _dbm_to_mw


RAW_FILES = {
    "mnist": ("MNIST", "Mnist_restored.npz"),
    "fmnist": ("Fashion-MNIST", "fashionmnist_restored.npz"),
    "emnist": ("EMNIST", "EMnist_restored.npz"),
}

HARD_FILES = {
    "fmnist": "fmnist_output_6.mat",
    "mnist": "mnist_output_4.mat",
    "emnist": "emnist_output_8.mat",
}


def _resolve_dir(data_dir, subdir):
    data_dir = Path(data_dir)
    nested = data_dir / subdir
    return nested if nested.exists() else data_dir


def _ensure_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_npz(path):
    data = np.load(path)
    if "images" not in data or "labels" not in data:
        raise KeyError(f"{path} must contain 'images' and 'labels'. Found keys: {data.files}")
    return data["images"], data["labels"]


def _load_feature_sample(path, num_samples, num_channels, sample_index):
    mat = io.loadmat(path, variable_names=["train_x", "labels"])
    if "train_x" not in mat or "labels" not in mat:
        raise KeyError(f"{path} must contain 'train_x' and 'labels'. Found keys: {list(mat.keys())}")

    sample_index = int(sample_index) % num_samples
    start = sample_index * FEATURE_MAP_PIXELS
    stop = start + FEATURE_MAP_PIXELS
    raw_x = mat["train_x"].astype(np.float32)
    feature = raw_x[start:stop].reshape(FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, num_channels).transpose(2, 0, 1)
    labels = np.asarray(mat["labels"]).reshape(-1)
    label = int(labels[sample_index]) if labels.size > sample_index else -1
    return feature, label


def _show_grid(images, titles, ncols, figsize, output_path, cmap="gray"):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, title in zip(axes.flat, images, titles):
        im = ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if cmap != "gray":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_raw(data_dir, output_dir, num_samples, sample_index, seed):
    raw_dir = _resolve_dir(data_dir, "raw_data")
    rng = np.random.default_rng(seed)

    images_to_plot = []
    titles = []
    for key, (display_name, filename) in RAW_FILES.items():
        images, labels = _load_npz(raw_dir / filename)
        if sample_index is None:
            count = min(num_samples, len(labels))
            indices = rng.choice(len(labels), size=count, replace=False)
        else:
            indices = np.arange(sample_index, sample_index + min(num_samples, len(labels))) % len(labels)
        for idx in indices:
            images_to_plot.append(images[idx])
            titles.append(f"{display_name}\ny={int(labels[idx])}")

    output_path = output_dir / "raw_samples.png"
    _show_grid(images_to_plot, titles, ncols=max(1, num_samples), figsize=(1.8 * max(1, num_samples), 5.6), output_path=output_path)
    print(f"Saved raw data visualization to {output_path}")


def visualize_hard(data_dir, output_dir, sample_index):
    hard_dir = Path(data_dir)
    images = []
    titles = []
    for task in TASKS:
        feature, label = _load_feature_sample(
            hard_dir / HARD_FILES[task.key],
            task.num_samples,
            task.hard_channels,
            sample_index,
        )
        for channel in range(task.hard_channels):
            images.append(feature[channel])
            titles.append(f"{task.display_name}\ny={label}, ch={channel + 1}")

    output_path = output_dir / "hard_pen_feature_maps.png"
    _show_grid(images, titles, ncols=8, figsize=(14, 7), output_path=output_path, cmap="viridis")
    print(f"Saved hard-routing PEN visualization to {output_path}")


def _weighted_sparse_18(feature, num_channels):
    unified = np.zeros((18, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE), dtype=np.float32)
    if num_channels == 4:
        unified[0:4] = feature
    elif num_channels == 6:
        unified[4:10] = feature
    elif num_channels == 8:
        unified[10:18] = feature
    else:
        raise ValueError(f"Unsupported channel count: {num_channels}")
    return unified


def visualize_weighted(data_dir, output_dir, sample_index):
    weighted_dir = Path(data_dir)
    task_images = []
    task_titles = []
    occupancy_rows = []
    occupancy_labels = []

    for task in TASKS:
        for num_channels in WEIGHTED_CHANNELS:
            feature, label = _load_feature_sample(
                weighted_dir / f"{task.key}_output_{num_channels}.mat",
                task.num_samples,
                num_channels,
                sample_index,
            )
            feature_mw = _dbm_to_mw(feature)
            unified = _weighted_sparse_18(feature_mw, num_channels)
            occupancy_rows.append((np.abs(unified).sum(axis=(1, 2)) > 0).astype(float))
            occupancy_labels.append(f"{task.display_name} {num_channels}ch")

            for channel in range(num_channels):
                task_images.append(feature_mw[channel])
                task_titles.append(f"{task.display_name} {num_channels}ch\ny={label}, ch={channel + 1}")

    feature_path = output_dir / "weighted_pen_feature_maps.png"
    _show_grid(task_images, task_titles, ncols=8, figsize=(14, 18), output_path=feature_path, cmap="viridis")
    print(f"Saved weighted-routing PEN visualization to {feature_path}")

    fig, ax = plt.subplots(figsize=(10, 4))
    occupancy = np.vstack(occupancy_rows)
    ax.imshow(occupancy, cmap="Greys", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(occupancy_labels)))
    ax.set_yticklabels(occupancy_labels, fontsize=8)
    ax.set_xticks(np.arange(18))
    ax.set_xticklabels([str(i + 1) for i in range(18)], fontsize=8)
    ax.set_xlabel("18-channel unified representation")
    ax.set_title("Weighted-routing channel occupancy")
    fig.tight_layout()
    occupancy_path = output_dir / "weighted_18_channel_occupancy.png"
    fig.savefig(occupancy_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved weighted-routing occupancy visualization to {occupancy_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize raw images and PMoE PEN feature maps.")
    parser.add_argument("--source", choices=("raw", "hard", "weighted", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "visualizations")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of raw images per dataset.")
    parser.add_argument("--sample-index", type=int, default=None, help="Optional starting sample index.")
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = _ensure_output_dir(args.output_dir)

    sample_index = 0 if args.sample_index is None else args.sample_index

    if args.source in ("raw", "all"):
        visualize_raw(args.data_dir, output_dir, args.num_samples, args.sample_index, args.seed)
    if args.source in ("hard", "all"):
        visualize_hard(args.data_dir, output_dir, sample_index)
    if args.source in ("weighted", "all"):
        visualize_weighted(args.data_dir, output_dir, sample_index)


if __name__ == "__main__":
    main()

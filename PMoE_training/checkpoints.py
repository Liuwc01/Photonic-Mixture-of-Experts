from pathlib import Path

import torch


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _rename_by_prefix(key, replacements):
    for old, new in replacements:
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def _map_hard_key(key):
    replacements = (
        ("adapter_fashion.", "adapters.0."),
        ("adapter_custom_mnist.", "adapters.1."),
        ("adapter_emnist.", "adapters.2."),
        ("shared_conv_block.", "shared_conv."),
        ("head_fashion.", "heads.0."),
        ("head_custom_mnist.", "heads.1."),
        ("head_emnist.", "heads.2."),
    )
    return _rename_by_prefix(key, replacements)


def _map_weighted_key(key):
    replacements = (
        ("bn_fmnist.", "adapters.0."),
        ("bn_mnist.", "adapters.1."),
        ("bn_emnist.", "adapters.2."),
        ("shared_backbone.", "shared_conv."),
        ("fc.", "shared_fc."),
        ("heads.fmnist.", "heads.0."),
        ("heads.mnist.", "heads.1."),
        ("heads.emnist.", "heads.2."),
    )
    return _rename_by_prefix(key, replacements)


def remap_pretrained_state_dict(state_dict, model, routing):
    target_state = model.state_dict()
    mapped = {}
    ignored = []
    shape_mismatches = []

    for key, value in state_dict.items():
        if key.endswith("total_ops") or key.endswith("total_params") or key in {"total_ops", "total_params"}:
            ignored.append(key)
            continue
        if routing == "hard":
            new_key = _map_hard_key(key)
        elif routing == "weighted":
            new_key = _map_weighted_key(key)
        else:
            raise ValueError(f"Unsupported routing mode: {routing}")

        if new_key not in target_state:
            ignored.append(key)
            continue
        if target_state[new_key].shape != value.shape:
            shape_mismatches.append((key, new_key, tuple(value.shape), tuple(target_state[new_key].shape)))
            continue
        mapped[new_key] = value

    missing = [key for key in target_state if key not in mapped]
    return mapped, missing, ignored, shape_mismatches


def load_pretrained_weights(model, checkpoint_path, routing, device="cpu", strict=True):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained weight file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    mapped, missing, ignored, shape_mismatches = remap_pretrained_state_dict(state_dict, model, routing)

    if shape_mismatches:
        details = "; ".join(
            f"{old}->{new}: checkpoint{old_shape} != model{new_shape}"
            for old, new, old_shape, new_shape in shape_mismatches
        )
        raise RuntimeError(f"Shape mismatches while loading {checkpoint_path}: {details}")

    if strict and missing:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} did not provide all model parameters. "
            f"Missing keys: {missing}. Ignored checkpoint keys: {ignored}"
        )

    load_info = model.load_state_dict(mapped, strict=False)
    print(
        f"Loaded {len(mapped)} tensors from {checkpoint_path}. "
        f"Missing model keys: {len(load_info.missing_keys)}; ignored checkpoint keys: {len(ignored)}."
    )
    return {
        "loaded_keys": sorted(mapped.keys()),
        "missing_keys": load_info.missing_keys,
        "unexpected_keys": load_info.unexpected_keys,
        "ignored_keys": ignored,
    }

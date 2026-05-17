from pathlib import Path
import random

import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config import (
        DEFAULT_CLIP_NORM,
        DEFAULT_ETA_MIN,
        DEFAULT_LR,
        DEFAULT_RESULTS_DIR,
        DEFAULT_WEIGHT_DECAY,
        TASKS,
    )
except ImportError:
    from config import (
        DEFAULT_CLIP_NORM,
        DEFAULT_ETA_MIN,
        DEFAULT_LR,
        DEFAULT_RESULTS_DIR,
        DEFAULT_WEIGHT_DECAY,
        TASKS,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_model(model, loaders, criterion, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    task_accs = []
    task_losses = []
    with torch.no_grad():
        for task_id, task in enumerate(TASKS):
            correct = 0
            samples = 0
            loss_sum = 0.0
            loader = loaders[task.key]
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device).long()
                outputs = model(images, task_id)
                loss = criterion(outputs, labels)
                batch_size = labels.size(0)
                loss_sum += loss.item() * batch_size
                correct += (outputs.argmax(1) == labels).sum().item()
                samples += batch_size
            task_accs.append(100.0 * correct / samples if samples else 0.0)
            task_losses.append(loss_sum / samples if samples else 0.0)
            total_correct += correct
            total_samples += samples
    total_acc = 100.0 * total_correct / total_samples if total_samples else 0.0
    return total_acc, task_accs, task_losses


def train_model(
    model,
    loaders,
    routing,
    epochs,
    device,
    results_dir=DEFAULT_RESULTS_DIR,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    eta_min=DEFAULT_ETA_MIN,
    clip_norm=DEFAULT_CLIP_NORM,
    max_batches=None,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    history = {
        "total_train_acc": [],
        "total_test_acc": [],
        "total_train_loss": [],
        "total_test_loss": [],
    }
    for task in TASKS:
        history[f"{task.key}_train_acc"] = []
        history[f"{task.key}_test_acc"] = []
        history[f"{task.key}_train_loss"] = []
        history[f"{task.key}_test_loss"] = []

    best_test_acc = 0.0
    best_snapshot = {}
    print(f"Training {routing}-routing PMoE on device: {device}")

    train_iters = [loaders[task.key]["train"] for task in TASKS]
    train_eval_loaders = {task.key: loaders[task.key]["train_eval"] for task in TASKS}
    test_loaders = {task.key: loaders[task.key]["test"] for task in TASKS}

    for epoch in range(1, epochs + 1):
        model.train()
        task_loss_sums = [0.0 for _ in TASKS]
        task_sample_sums = [0 for _ in TASKS]

        for batch_idx, batches in enumerate(zip(*train_iters), start=1):
            optimizer.zero_grad()
            combined_loss = 0.0
            for task_id, batch in enumerate(batches):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device).long()
                outputs = model(images, task_id)
                loss = criterion(outputs, labels)
                combined_loss = combined_loss + loss
                task_loss_sums[task_id] += loss.item() * labels.size(0)
                task_sample_sums[task_id] += labels.size(0)
            avg_loss = combined_loss / len(TASKS)
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            if max_batches is not None and batch_idx >= max_batches:
                break

        scheduler.step()

        train_acc, task_train_accs, task_train_losses = evaluate_model(
            model, train_eval_loaders, criterion, device
        )
        test_acc, task_test_accs, task_test_losses = evaluate_model(model, test_loaders, criterion, device)

        history["total_train_acc"].append(train_acc)
        history["total_test_acc"].append(test_acc)
        history["total_train_loss"].append(float(np.mean(task_train_losses)))
        history["total_test_loss"].append(float(np.mean(task_test_losses)))
        for idx, task in enumerate(TASKS):
            history[f"{task.key}_train_acc"].append(task_train_accs[idx])
            history[f"{task.key}_test_acc"].append(task_test_accs[idx])
            history[f"{task.key}_train_loss"].append(task_train_losses[idx])
            history[f"{task.key}_test_loss"].append(task_test_losses[idx])

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_snapshot = {
                "epoch": epoch,
                "best_total_test_acc": test_acc,
                "corresponding_total_train_acc": train_acc,
                "task_test_accs": np.array(task_test_accs),
                "task_train_accs": np.array(task_train_accs),
            }
            torch.save(model.state_dict(), results_dir / f"{routing}_best_model.pt")

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch}/{epochs} | Test Acc: {test_acc:.2f}% | "
                f"Best Acc: {best_test_acc:.2f}% | Train Acc: {train_acc:.2f}%"
            )

    result = {
        "routing": routing,
        "epoch_history": {key: np.array(value) for key, value in history.items()},
        "best_snapshot": best_snapshot,
        "task_order": np.array([task.key for task in TASKS], dtype=object),
    }
    io.savemat(results_dir / f"{routing}_training_results.mat", result)
    torch.save(model.state_dict(), results_dir / f"{routing}_final_model.pt")
    return result

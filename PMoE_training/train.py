import argparse
from pathlib import Path

import torch

try:
    from .checkpoints import load_pretrained_weights
    from .config import DEFAULT_DATA_DIR, DEFAULT_EPOCHS, DEFAULT_RESULTS_DIR, DEFAULT_ROUTING_SEEDS, DEFAULT_WEIGHT_FILES, TASKS
    from .data import build_hard_routing_loaders, build_weighted_routing_loaders
    from .models import HardRoutingPMoE, WeightedRoutingPMoE
    from .trainer import evaluate_model, set_seed, train_model
except ImportError:
    from checkpoints import load_pretrained_weights
    from config import DEFAULT_DATA_DIR, DEFAULT_EPOCHS, DEFAULT_RESULTS_DIR, DEFAULT_ROUTING_SEEDS, DEFAULT_WEIGHT_FILES, TASKS
    from data import build_hard_routing_loaders, build_weighted_routing_loaders
    from models import HardRoutingPMoE, WeightedRoutingPMoE
    from trainer import evaluate_model, set_seed, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train hard-routing or weighted-routing PMoE models.")
    parser.add_argument("--routing", choices=("hard", "weighted"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to the original notebook seed for each routing mode.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional smoke-test limit per epoch.")
    parser.add_argument("--weights", type=Path, default=None, help="Optional pretrained weight file to load.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load the default pretrained weights for the selected routing mode.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Evaluate loaded weights without training.")
    parser.add_argument(
        "--non-strict-weights",
        action="store_true",
        help="Allow missing model keys when loading pretrained weights.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = DEFAULT_ROUTING_SEEDS[args.routing]
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.routing == "hard":
        loaders = build_hard_routing_loaders(args.data_dir, seed=args.seed)
        model = HardRoutingPMoE()
    else:
        loaders = build_weighted_routing_loaders(args.data_dir, seed=args.seed)
        model = WeightedRoutingPMoE()

    weight_path = args.weights
    if args.pretrained:
        weight_path = DEFAULT_WEIGHT_FILES[args.routing]
    if weight_path is not None:
        load_pretrained_weights(
            model,
            weight_path,
            routing=args.routing,
            device=device,
            strict=not args.non_strict_weights,
        )

    if args.eval_only:
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        test_loaders = {task.key: loaders[task.key]["test"] for task in TASKS}
        total_acc, task_accs, task_losses = evaluate_model(model, test_loaders, criterion, device)
        print(f"Evaluation accuracy: {total_acc:.2f}%")
        for task, acc, loss in zip(TASKS, task_accs, task_losses):
            print(f"  {task.display_name}: acc={acc:.2f}% loss={loss:.4f}")
        return

    train_model(
        model=model,
        loaders=loaders,
        routing=args.routing,
        epochs=args.epochs,
        device=device,
        results_dir=args.results_dir,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()

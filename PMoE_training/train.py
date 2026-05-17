import argparse
from pathlib import Path

import torch

try:
    from .config import DEFAULT_DATA_DIR, DEFAULT_EPOCHS, DEFAULT_RESULTS_DIR, DEFAULT_SEED
    from .data import build_hard_routing_loaders, build_weighted_routing_loaders
    from .models import HardRoutingPMoE, WeightedRoutingPMoE
    from .trainer import set_seed, train_model
except ImportError:
    from config import DEFAULT_DATA_DIR, DEFAULT_EPOCHS, DEFAULT_RESULTS_DIR, DEFAULT_SEED
    from data import build_hard_routing_loaders, build_weighted_routing_loaders
    from models import HardRoutingPMoE, WeightedRoutingPMoE
    from trainer import set_seed, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train hard-routing or weighted-routing PMoE models.")
    parser.add_argument("--routing", choices=("hard", "weighted"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional smoke-test limit per epoch.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.routing == "hard":
        loaders = build_hard_routing_loaders(args.data_dir, seed=args.seed)
        model = HardRoutingPMoE()
    else:
        loaders = build_weighted_routing_loaders(args.data_dir, seed=args.seed)
        model = WeightedRoutingPMoE()

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

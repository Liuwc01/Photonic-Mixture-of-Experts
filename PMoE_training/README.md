# PMoE Training Code

The code reads PEN feature maps from `.mat` files and trains the shared electronic backend.

## Files

- `config.py`: task metadata, training hyperparameters, and default paths.
- `data.py`: `.mat` loading, label normalization, train/test splitting, and DataLoader construction.
- `models.py`: `HardRoutingPMoE` and `WeightedRoutingPMoE`.
- `trainer.py`: training, evaluation, scheduler, gradient clipping, and result saving.
- `train.py`: command-line entry point.

## Data Layout

By default, the scripts read data from `../exp_data/`:

- `fmnist_output_4.mat`, `fmnist_output_6.mat`, `fmnist_output_8.mat`
- `mnist_output_4.mat`, `mnist_output_6.mat`, `mnist_output_8.mat`
- `emnist_output_4.mat`, `emnist_output_6.mat`, `emnist_output_8.mat`

Each file is expected to contain `train_x` and `labels`.

## Run

```bash
python pmoe_refactored/train.py --routing hard
python pmoe_refactored/train.py --routing weighted
```

For a quick smoke test:

```bash
python pmoe_refactored/train.py --routing hard --epochs 1 --max-batches 1
python pmoe_refactored/train.py --routing weighted --epochs 1 --max-batches 1
```

Outputs are saved to `pmoe_refactored/results/` by default.

## Model Notes

- Hard-routing uses one PEN output per task: 6-channel Fashion-MNIST, 4-channel MNIST, and 8-channel EMNIST. Task-specific BatchNorm is followed by zero-padding to 8 channels.
- Weighted-routing combines the 4-, 6-, and 8-channel PEN outputs for each sample into an 18-channel representation, then applies a `1x1 Conv2d(18, 8)` projection.
- Both models use the same shared backend and training protocol: 300 epochs, AdamW, learning rate `8e-5`, weight decay `1e-4`, cosine annealing, and gradient clipping.

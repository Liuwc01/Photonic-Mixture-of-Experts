# Photonic Mixture-of-Experts Supplementary Package

This repository contains the supplementary package accompanying the manuscript:

**Photonic Mixture-of-Experts for scalable on-chip multi-task optical neural networks**

![PMoE processing pipeline](assets/pmoe_pipeline.jpg)

The package provides the raw dataset subsets used in the manuscript, the experimental input and output data for the photonic expert networks (PENs), PMoE network construction and training code, pretrained/output results, and the source data used to generate figures in the manuscript and supplementary materials.

## Repository Structure

```text
.
+-- README.md
+-- DATA.md
+-- CITATION.cff
+-- requirements.txt
+-- assets/
+-- multi_datasets/
+-- flattened_input/
+-- exp_data/
+-- PMoE_training/
+-- source_data/
```

Additional data details are provided in `DATA.md`. Citation metadata is provided in `CITATION.cff`. The `assets/` folder is reserved for overview figures and processing schematics, such as a PMoE pipeline diagram.

### `multi_datasets/`

Selected image subsets used in this work:

- `Mnist_2000.npz`: 2,000 MNIST samples.
- `fashionmnist_2000.npz`: 2,000 Fashion-MNIST samples.
- `EMnist_5200.npz`: 5,200 Extended-MNIST samples.

These subsets are used to construct the multi-domain PMoE training and evaluation protocol.

### `flattened_input/`

Experimental input data prepared for the optical/PEN experiments:

- `mnist_input_9.mat`
- `fmnist_input_9.mat`
- `emnist_input_9.mat`

These files contain the flattened input representations used for the photonic expert network experiments.

### `exp_data/`

Experimental PEN output feature maps used by the released PMoE training code:

- `mnist_output_4.mat`, `mnist_output_6.mat`, `mnist_output_8.mat`
- `fmnist_output_4.mat`, `fmnist_output_6.mat`, `fmnist_output_8.mat`
- `emnist_output_4.mat`, `emnist_output_6.mat`, `emnist_output_8.mat`

For weighted-routing PMoE training, all nine PEN output files are used. For hard-routing PMoE training, the code uses the task-specific PEN outputs `mnist_output_4.mat`, `fmnist_output_6.mat`, and `emnist_output_8.mat`.

### `PMoE_training/`

Training and evaluation code for the PMoE backend:

- `models.py`: hard-routing and weighted-routing PMoE model definitions.
- `data.py`: data loading and train/test split construction from `exp_data/`.
- `trainer.py`: offline training loop, evaluation, optimizer, scheduler, and result saving.
- `train.py`: command-line entry point for training and evaluation.
- `checkpoints.py`: utilities for loading pretrained/default weights.
- `visualize_data.py`: utilities for visualizing PEN feature maps.
- `weights/`: default pretrained weights.
- `results/`: saved training results and model checkpoints.
- `visualizations/`: generated example visualizations.

### `source_data/`

Source data for manuscript and supplementary figures. The folder contains Excel files organized by figure, including `fig_2/`, `fig_3/`, `fig_4/`, `fig_5/`, and `sup_figs/`.

## PMoE Training Code

In the full research workflow, multiple photonic expert networks (PENs) and the shared digital/electronic network (SDN) form the PMoE system. The PENs serve as the optical front end for routed and weighted feature processing, while the SDN performs shared backend classification.

In this released supplementary package, the `exp_data/` folder provides fixed PEN-generated experimental output feature maps without routing weights already applied. Therefore, users can directly train the backend PMoE/SDN models from the provided PEN experimental outputs. The released training code keeps the main PMoE architecture intact and adapts the data-loading pipeline so both hard-routing and weighted-routing variants can be trained from the supplied PEN output data.

The training protocol follows the balanced multi-domain setup used in the manuscript:

- 2,000 MNIST samples, 2,000 Fashion-MNIST samples, and 5,200 Extended-MNIST samples.
- A 4:1 train/test split.
- Weighted batch sizes of `10:10:26` for MNIST, Fashion-MNIST, and Extended-MNIST, respectively, so that each domain contributes a balanced number of iterations within one epoch.
- Default training for 300 epochs using AdamW, cosine annealing learning-rate scheduling, and gradient clipping.

## Usage

Run commands from the repository root.

Install the Python dependencies in an environment with PyTorch support:

```bash
pip install -r requirements.txt
```

Train the hard-routing PMoE model:

```bash
python -m PMoE_training.train --routing hard
```

Train the weighted-routing PMoE model:

```bash
python -m PMoE_training.train --routing weighted
```

Evaluate the default pretrained hard-routing weights:

```bash
python -m PMoE_training.train --routing hard --pretrained --eval-only
```

Evaluate the default pretrained weighted-routing weights:

```bash
python -m PMoE_training.train --routing weighted --pretrained --eval-only
```

Common optional arguments:

- `--data-dir`: path to the PEN output data folder. Defaults to `exp_data/`.
- `--results-dir`: path for saving training outputs. Defaults to `PMoE_training/results/`.
- `--epochs`: number of training epochs. Defaults to `300`.
- `--device`: training device, such as `cuda` or `cpu`.
- `--weights`: path to a custom weight file to load.

For quick smoke tests, `--max-batches` can be used to limit the number of batches per epoch.

## Citation

If you use this repository, please cite the associated manuscript:

**Photonic Mixture-of-Experts for scalable on-chip multi-task optical neural networks**

Citation metadata is provided in `CITATION.cff`. The author list, DOI, and publication venue can be updated after publication.

## License

The supplementary data, documentation, and figure source data are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0). Code in `PMoE_training/` is released for research and reproducibility use. The MNIST, Fashion-MNIST, and EMNIST subsets retain the terms of their original dataset sources.

## Notes

The released training scripts operate on PEN-generated feature maps rather than optimizing optical diffraction parameters end-to-end from raw images. This makes the package suitable for reproducing and inspecting backend PMoE training using the experimental PEN outputs provided with the repository.

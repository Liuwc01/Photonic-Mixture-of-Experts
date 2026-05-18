# Data Description

This document summarizes the data files released with the supplementary package for **Photonic Mixture-of-Experts for scalable on-chip multi-task optical neural networks**.

## Dataset Subsets

The `multi_datasets/` folder contains the image subsets used in the multi-domain experiments:

- `Mnist_2000.npz`: 2,000 MNIST samples.
- `fashionmnist_2000.npz`: 2,000 Fashion-MNIST samples.
- `EMnist_5200.npz`: 5,200 Extended-MNIST samples.

These files provide the fixed dataset subsets used to construct the PMoE training and evaluation protocol.

## Optical Experiment Inputs

The `flattened_input/` folder contains input arrays prepared for the photonic expert network (PEN) experiments:

- `mnist_input_9.mat`
- `fmnist_input_9.mat`
- `emnist_input_9.mat`

These files are the flattened input representations used for the optical experiments.

## PEN Experimental Outputs

The `exp_data/` folder contains PEN-generated experimental output feature maps:

- `mnist_output_4.mat`, `mnist_output_6.mat`, `mnist_output_8.mat`
- `fmnist_output_4.mat`, `fmnist_output_6.mat`, `fmnist_output_8.mat`
- `emnist_output_4.mat`, `emnist_output_6.mat`, `emnist_output_8.mat`

Each `.mat` file is expected by the released training code to contain:

- `train_x`: PEN output feature maps arranged as flattened spatial features.
- `labels`: class labels corresponding to the original image samples.

The PMoE training code reshapes `train_x` into `26 x 26` feature maps. For a file with `C` PEN output channels, the expected feature tensor has `C` channels after reshaping.

## Hard-Routing Inputs

Hard-routing PMoE training uses one task-specific PEN output per domain:

- MNIST: `mnist_output_4.mat`
- Fashion-MNIST: `fmnist_output_6.mat`
- Extended-MNIST: `emnist_output_8.mat`

The hard-routing model applies task-specific normalization and zero-padding so that all task inputs are standardized to 8 channels before entering the shared backend network.

## Weighted-Routing Inputs

Weighted-routing PMoE training uses all nine PEN output files. For each original image sample, the corresponding 4-channel, 6-channel, and 8-channel PEN outputs are fused into one 18-channel representation:

```text
channels 0:4    <- *_output_4.mat
channels 4:10   <- *_output_6.mat
channels 10:18  <- *_output_8.mat
```

The same original image index is kept together across the 4-channel, 6-channel, and 8-channel outputs. The train/test split is performed at the original-image level, preventing same-source PEN variants from being split across training and testing.

## Figure Source Data

The `source_data/` folder contains source data for manuscript and supplementary figures. Files are organized by figure folder, including `fig_2/`, `fig_3/`, `fig_4/`, `fig_5/`, and `sup_figs/`.

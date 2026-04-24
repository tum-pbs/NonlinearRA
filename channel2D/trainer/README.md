# 2D Transitional Channel Flow — Neural Network Emulator

This repository contains the code for training and analyzing neural network (NN) emulators of **2D plane Poiseuille flow** (Re = 2000), as described in the paper. The emulator learns the temporal evolution of infinitesimal velocity perturbations around the base parabolic profile $U(y) = 1 - y^2$, and is subsequently used for data-driven **stability analysis** and **resolvent analysis** via the Jacobian of the trained network.

---

## Overview

The 2D channel flow example demonstrates how a neural network emulator — trained purely on snapshot data — can recover the linear stability properties of the underlying PDE operator (the Orr–Sommerfeld operator), including:

- **Eigenspectrum** (A-branch, P-branch, S-branch modes)
- **Eigenvectors** (spatial structures of wall modes and center modes)
- **Resolvent gain curves** and **optimal forcing/response modes**

Two dataset regimes are supported:
- **Weakly nonlinear** ($\epsilon = 10^{-6}$): perturbations remain in the linear regime throughout the transient.
- **Nonlinear** ($\epsilon = 10^{-1}$): strong nonlinear effects are present during the transient phase.

Two approaches are provided (physical space and SVD-reduced space):
- **U-Net** (`train_channel_UNet.py`): operates directly on the 2D velocity perturbation fields $(u, v)$.
- **MLP on SVD-reduced coordinates** (`train_channel_UNet_svd.py`): projects data onto the leading $r$ SVD modes and trains an MLP in the reduced space. **All filenames related to SVD-reduced approach have a suffix "*_svd".**

---

## Features

- **U-Net emulator** with configurable depth, channel width, and multi-step temporal unrolling.
- **SVD-based dimensionality reduction** for training an MLP on a low-dimensional latent space.
- **Jacobian extraction** of the trained emulator at the base flow, enabling linear stability analysis.
- **Stability analysis**: eigenspectrum computation and comparison against the Orr–Sommerfeld ground truth.
- **Resolvent analysis**: optimal gain curves and forcing/response mode shapes via DMD-projected resolvent.
- Support for **ensemble averaging** over multiple independent training runs.

---

## Repository Structure

```
.
├── trainer/
│   ├── train_channel_UNet.py        # Train U-Net emulator on full 2D fields
│   ├── prediction.py                # Rollout predictions and error visualization
│   ├── stability_analysis.py        # Jacobian extraction and eigenspectrum computation
│   ├── resolvent_analysis.py        # Resolvent gain curves and optimal mode extraction
│   ├── stability_truth.mat          # Ground truth of stability analysis
│   ├── resolvent_truth.mat          # Ground truth of resolvent analysis
│   └── ...
├── dataset/                     # Training data (download from HuggingFace, see below)
│   ├── random1/
│   ├── random2/
│   └── ...
└── results/                     # All outputs are saved here (auto-created)
```

---

## Dataset

The training dataset consists of **60 trajectories**, each containing **400 snapshots** of the 2D velocity perturbation field $(u, v)$, saved every 0.5 time units. The spatial domain is $2\pi \times 2$, discretized on a $32 \times 64$ grid (Fourier in $x$, Chebyshev in $y$). The numerical computation is based on a spectral solver [shenfun](https://github.com/spectralDNS/shenfun).

**Download [dataset](https://huggingface.co/datasets/jiatingdaole1109/NonlinearRA/tree/main) from HuggingFace and place into the `dataset/` folder:**

---

## Usage

### 1. Train the neural Emulator

#### Unet version
```bash
python train_channel_UNet.py \
    --expo 5 \
    --level 3 \
    --batch_size 64 \
    --n_rollout 10 \
    --n_iterations 500000 \
    --n_runs 5 \
    --gpu 0
```

| Argument | Default | Description |
|---|---|---|
| `--expo` | `5` | Log₂ of the number of hidden channels in the top U-Net layer (e.g., `5` → 32 channels) |
| `--level` | `3` | Number of encoding/decoding levels in the U-Net |
| `--batch_size` | `64` | Training batch size |
| `--n_rollout` | `10` | Number of temporal unrolling steps during training |
| `--n_iterations` | `500000` | Total number of gradient update steps |
| `--n_runs` | `5` | Number of independent training runs (for ensemble Jacobian) |
| `--gpu` | `'0'` | CUDA device ID |

**Outputs** (saved to `results/expo{expo}_level{level}_batch{batch_size}_roll{n_rollout}_nrun{n_runs}/`):
- `run{i}/channel_UNet.eqx` — serialized model weights
- `run{i}/loss_channel_UNet.png` — training loss curve
- `norm_dynamics.npz` — normalization statistics

---

#### SVD-Reduced MLP version

Uses SVD to project the data onto `r` leading modes, then trains an MLP in this reduced space.

```bash
python train_channel_UNet_svd.py \
    --expo 7 \
    --layer 4 \
    --batch_size 64 \
    --n_rollout 10 \
    --n_iterations 500000 \
    --r 20 \
    --n_runs 5 \
    --gpu 0
```

| Argument | Default | Description |
|---|---|---|
| `--expo` | `7` | Log₂ of the MLP hidden layer width |
| `--layer` | `4` | Number of hidden layers in the MLP |
| `--batch_size` | `64` | Training batch size |
| `--n_rollout` | `10` | Number of temporal unrolling steps during training |
| `--n_iterations` | `500000` | Total number of gradient update steps |
| `--r` | `20` | Number of SVD modes retained for dimensionality reduction |
| `--n_runs` | `5` | Number of independent training runs |
| `--gpu` | `'0'` | CUDA device ID |

**Outputs** (saved to `results/expo{expo}_layer{layer}_batch{batch_size}_roll{n_rollout}_nrun{n_runs}_svd{r}/`):
- `run{i}/channel_MLP.eqx` — serialized MLP model weights
- `run{i}/loss_channel_MLP.png` — training loss curve
- `norm_dynamics_svd.npz` — normalization statistics and SVD basis ($U$, $\Sigma$, $V^H$)

---

### 2. Predict and Visualize Rollouts

```bash
python prediction.py \
    --savedir ../results/expo5_level3_batch16_roll10_nrun5
```

Performs an auto-regressive rollout over the test trajectory and saves:
- `run{i}/frames_channel_UNet/frame_*.png` — per-frame comparison plots (ground truth, prediction, difference)
- `run{i}/video_channel_UNet.mp4` — animation of the rollout
- `run{i}/predict_loss.npy` / `predict_loss.png` — relative $L^1$ error over time for $u$ and $v$

---

### 3. Stability Analysis (Eigenspectrum)

Computes the Jacobian of the trained emulator at the zero base flow state, extracts eigenvalues, and compares against ground truth from the Orr–Sommerfeld operator.

```bash
python stability_analysis.py \
    --savedir ../results/expo5_level3_batch16_roll10_nrun5
```

**Requires:** `stability_truth.mat` (ground truth eigenvalues, place in the working directory).

**Outputs:**
- `run{i}/jacobian.npy` — per-run Jacobian matrix
- `ensemble_jacobian.npy` — ensemble-averaged Jacobian
- `eigenspectrum.png` — eigenvalue comparison (continuous-time and discrete-time spectra)
- `eigenvectors.png` — spatial structure of the leading modes

---

### 4. Resolvent Analysis

Uses the DMD-projected resolvent to compute optimal gain curves and forcing/response modes.

```bash
python resolvent_analysis.py \
    --savedir ../results/expo5_level3_batch16_roll10_nrun5
```

**Requires:**
- `ensemble_jacobian.npy` (produced by `stability_analysis.py`)
- `resolvent_truth.mat` (ground truth resolvent modes, place in the working directory)

**Outputs:**
- `resolvent_results.npz` — eigenvalues, singular vectors, and gain curves
- `singular_vectors.png` — optimal response and forcing mode spatial structures
- `gain_distribution.png` — resolvent gain curves vs. frequency $\omega$

---

## Recommended Workflow

```
Download dataset
       ↓
train_channel_UNet.py   (or train_channel_UNet_svd.py)
       ↓
prediction.py           (optional: check rollout accuracy)
       ↓
stability_analysis.py   (computes Jacobian + eigenspectrum)
       ↓
resolvent_analysis.py   (computes gain curves + resolvent modes)
```

---

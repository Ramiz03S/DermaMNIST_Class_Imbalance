# Milestone 1 — Reproducing Published Baselines

Before studying diffusion-based augmentation for class imbalance on DermaMNIST, we reproduce the two reference results the project builds on: the MedMNIST v2 ResNet-18 classifier and the DDIM CIFAR-10 FID.

| Part | Goal | Notebook | 
|:---:|---|:---:|
| **A** | ResNet-18 on DermaMNIST | [![Notebook](https://img.shields.io/badge/Notebook-milestone1__a-2b3137?style=for-the-badge&logo=jupyter&logoColor=F37626)](milestone1_a.ipynb) | 
| **B** | DDIM sampling and FID on CIFAR-10 | [![Notebook](https://img.shields.io/badge/Notebook-milestone1__b-2b3137?style=for-the-badge&logo=jupyter&logoColor=F37626)](ddim_paper_milestone1_b.ipynb) | 

**Summary**

| Part | Metric | Ours | Paper |
|---|---|---|---|
| A | Test AUC / ACC | 0.914 ± 0.006 / 0.740 ± 0.016 | 0.917 / 0.735 |
| B | FID (legacy TF, 50 steps) | 4.72 | 4.67 |

Both notebooks were run on Colab with a T4 GPU and write their outputs to `MyDrive/deep_learning_project/`.

---

## Part A — ResNet-18 on DermaMNIST

Reproduces the MedMNIST v2 baseline for ResNet-18 at 28×28 on DermaMNIST.

### Method

- Uses the official [MedMNIST experiments](https://github.com/MedMNIST/experiments) script `MedMNIST2D/train_and_eval_pytorch.py` with `--data_flag=dermamnist`.
- Default settings: ResNet-18, 28×28 input, 100 epochs.
- The checkpoint with the best validation AUC is evaluated on the train, validation and test splits.

### Results

| Run | Test AUC | Test ACC |
|---|---|---|
| Run 0 | 0.916 | 0.724 |
| Run 1 | 0.919 | 0.755 |
| Run 2 | 0.908 | 0.741 |
| | 0.914 ± 0.006 | 0.740 ± 0.016 | 
| Paper (MedMNIST v2) | 0.917 | 0.735 |

Across three runs, ResNet-18 reaches a test AUC of 0.914 ± 0.006 and a test accuracy of 0.740 ± 0.016, and both published values (0.917 and 0.735) lie within one standard deviation of these means, so the MedMNIST v2 baseline is reproduced.

### Running

Run all cells (~45 min of training). Outputs (checkpoint, logs, per-split predictions) go to `Milestone_1_a/`.

---

## Part B — DDIM sampling and FID on CIFAR-10

Reproduces the DDIM paper's CIFAR-10 result: **FID 4.67 with 50 sampling steps**.

### Method

1. Load a pretrained DDPM CIFAR-10 U-Net and its beta schedule through `diffusers`.
2. Generate 50,000 images with 50 deterministic DDIM steps (η = 0), in 200 batches of 250. Batch *i* uses seed `1000 + i`, and each batch is saved to Drive as `.npy`, so a disconnected run resumes from the last saved batch.
3. Convert the batches to PNG on the local Colab disk.
4. Compute FID against the CIFAR-10 training set with `clean-fid`, in both `clean` and `legacy_tensorflow` modes.

### Implementation choices

**Quadratic timestep spacing.** The `diffusers` `DDIMScheduler` only supports linear spacings, while the paper's CIFAR-10 results use quadratic spacing, which puts more steps at low noise levels. The notebook therefore samples with `generalized_steps` from the [official DDIM repository](https://github.com/ermongroup/ddim), using the paper's sequence:

```python
seq = [int(s) for s in np.linspace(0, np.sqrt(1000 * 0.8), 50) ** 2]
```

**EMA weights.** The `google/ddpm-cifar10-32` checkpoint holds raw (non-EMA) weights, whereas the DDPM and DDIM papers evaluate with EMA weights. This is documented in a [Hugging Face discussion](https://huggingface.co/google/ddpm-cifar10-32/discussions/10), where users report FID ≈ 12 with this checkpoint and the Diff-Pruning authors share the [official EMA checkpoint converted to diffusers](https://github.com/VainF/Diff-Pruning/releases/download/v0.0.1/ddpm_ema_cifar10.zip). Both are evaluated with the same procedure; choose one with `weights_origin = "ema"` or `"google"`.

### Results

50,000 samples, 50 DDIM steps, quadratic spacing, compared against CIFAR-10 train:

| Weights | FID (clean) | FID (legacy TF) |
|---|---|---|
| EMA (converted official) | 5.59 | **4.72** |
| `google/ddpm-cifar10-32` (non-EMA) | 11.89 | 9.79 |
| **DDIM paper** | – | **4.67** |

With EMA weights, the legacy FID is within 0.05 of the paper. The remaining gap is consistent with sampling noise (a different set of seeds) and 8-bit quantization of the samples. The non-EMA checkpoint scores roughly twice as high, matching the reports above.

**Clean vs legacy FID.** `legacy_tensorflow` reproduces the original TensorFlow Inception pipeline used by the DDPM/DDIM papers, so it is the number to compare with 4.67. `clean` resizes to 299×299 with proper antialiasing; it is a more correct measurement but is not comparable to published legacy numbers.

### Running

Generation resumes automatically after a disconnect; FID takes about 4 min per mode. Outputs go to `Milestone_1_b/{ema,google}/run_0/ddim50_batches`.

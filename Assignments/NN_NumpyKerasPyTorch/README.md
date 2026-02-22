# CMPE 258 — Assignment: Neural Networks with NumPy, PyTorch & TensorFlow

**Course:** CMPE 258 Deep Learning | Spring 2026 | San José State University  
**Instructor:** Vijay Eranti  
**Author:** Parth   
**Video Walkthrough:** [Link](https://drive.google.com/drive/folders/1a9GvJspztK-fBFr7CX05-trZ9EiHxemv?usp=sharing)

---

## Overview

This repository contains **5 Google Colab notebooks** implementing the same 3-hidden-layer deep neural network for **non-linear regression** across multiple frameworks and abstraction levels. Each notebook trains on the same synthetic 3-variable dataset and produces comparable results, demonstrating how the same architecture maps to different implementations.

### Target Equation (3 variables)

```
y = 3·x₁² + 2·sin(π·x₂)·x₃ − x₃³ + 0.5·x₁·x₂ + 1
```

### Network Architecture (shared across all notebooks)

```
Input(3) → Dense(64, ReLU) → Dense(32, Tanh) → Dense(16, ReLU/Sigmoid) → Dense(1, Linear)
```

- **3 hidden layers** (extending the reference colab's 1-2 layers)
- **3 input variables** (extending the reference's 2 variables)
- **4D visualization** using PCA/t-SNE dimensionality reduction from scikit-learn

---

## Files

| File | Description |
|------|-------------|
| `colab_a_numpy_from_scratch.ipynb` | NumPy + tf.einsum — manual backprop, chain rule |
| `colab_b_pytorch_from_scratch.ipynb` | PyTorch raw tensors — no nn.Linear, no optimizer classes |
| `colab_c_pytorch_classes.ipynb` | PyTorch nn.Module — built-in layers + Adam optimizer |
| `colab_d_pytorch_lightning.ipynb` | PyTorch Lightning — LightningModule + Trainer |
| `colab_e_tensorflow_variants.ipynb` | TensorFlow 3-in-1: (i) low-level, (ii) Sequential, (iii) Functional |
| `README.md` | This file |
| `walkthrough.txt` | Quick walkthrough of each notebook (< 5 min each) |

---

## Detailed Notebook Descriptions

### Colab A — NumPy + tf.einsum (From Scratch)

**Key requirements met:**
- ✅ 3-layer deep neural network (vs reference colab's 1-2 layers)
- ✅ `tf.einsum` for all matrix multiplications (replaces `@` / `np.matmul`)
- ✅ 3 input variables (vs reference's 2)
- ✅ Manual backpropagation with chain rule
- ✅ Non-linear activations: ReLU, Tanh, Sigmoid
- ✅ 4D visualization with PCA and t-SNE (scikit-learn)

**Architecture:** Input(3) → Linear(64)+ReLU → Linear(32)+Tanh → Linear(16)+Sigmoid → Linear(1)

Each building block (LinearLayer, ReLU, Tanh, Sigmoid, MSELoss) implements both `__call__` (forward) and `backward` (gradient computation). The `LinearLayer` uses `tf.einsum('ij,jk->ik', A, B)` for all matrix products.

### Colab B — PyTorch From Scratch (No Built-in Layers)

**What "from scratch" means here:**
- Weights initialized with raw `torch.randn` + He initialization
- Forward pass uses `X @ W + b` — no `nn.Linear`
- Activations via `torch.relu` / `torch.tanh` — no `nn.ReLU`
- Backward pass via `loss.backward()` (PyTorch autograd)
- Parameter update via manual SGD in `torch.no_grad()` block — no `torch.optim`

This shows the intermediate level: raw tensor operations but leveraging PyTorch's automatic differentiation.

### Colab C — PyTorch Class-Based (Built-in Modules)

Standard PyTorch idioms:
- `nn.Module` subclass with `nn.Sequential`
- `nn.Linear`, `nn.ReLU`, `nn.Tanh` layers
- `nn.MSELoss` and `torch.optim.Adam`
- Proper `model.train()` / `model.eval()` discipline
- `nn.init.kaiming_normal_` weight initialization

### Colab D — PyTorch Lightning

Refactors Colab C into Lightning abstractions:
- `LightningModule` — encapsulates model + training_step + configure_optimizers
- `LightningDataModule` — handles data loading with train/val split
- `Trainer` — epoch loop, GPU, early stopping, logging
- `save_hyperparameters()` for reproducibility

### Colab E — TensorFlow Variants (3-in-1)

Three implementations in one notebook:

| Part | API Level | Key Feature |
|------|-----------|-------------|
| **(i)** | `tf.Variable` + `tf.GradientTape` | Full manual control, no Keras |
| **(ii)** | `tf.keras.Sequential` | Standard Keras with `model.fit()` |
| **(iii)** | `tf.keras.functional` | `Input()` → layer chaining → `Model()` |

Includes a side-by-side comparison of all three variants at the end.

---

## How to Run

1. Open any `.ipynb` file in [Google Colab](https://colab.research.google.com/)
2. Select **Runtime → Run all** (or run cells sequentially)
3. GPU is optional but recommended for faster training
4. All dependencies are pre-installed in Colab (numpy, torch, tensorflow, sklearn, matplotlib)
5. For Colab D, `pytorch-lightning` is installed via `!pip install -q pytorch-lightning`

---

## Comparison Table

| Feature | Colab A | Colab B | Colab C | Colab D | Colab E |
|---------|---------|---------|---------|---------|---------|
| Framework | NumPy+tf.einsum | PyTorch raw | PyTorch nn | Lightning | TensorFlow |
| Matmul | `tf.einsum` | `@` | `nn.Linear` | `nn.Linear` | varies |
| Backprop | Manual chain rule | Autograd | Autograd | Autograd | GradientTape/Keras |
| Optimizer | Manual SGD | Manual SGD | `Adam` | `Adam` | Manual/Adam |
| Abstraction | Lowest | Low | Medium | High | Low→High |
| Lines of code | ~120 | ~60 | ~40 | ~70 | ~150 (3 variants) |

---

## References

- **Reference Colab:** `deep_learning_fundamentals_part1.ipynb` (class notebook by Vijay Eranti)
- **PyTorch docs:** https://pytorch.org/docs/stable/
- **TensorFlow docs:** https://www.tensorflow.org/api_docs
- **PyTorch Lightning:** https://lightning.ai/docs/pytorch/stable/
- **scikit-learn PCA:** https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- **tf.einsum:** https://www.tensorflow.org/api_docs/python/tf/einsum

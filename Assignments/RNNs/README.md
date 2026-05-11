# 🧠 Sequence Models, Graphs & Vision Transformers — Deep Learning Tutorials

> **Topics:** RNNs & Beyond  
> [**Video Walkthrough**](https://drive.google.com/drive/folders/1FrlnZEKTrMKE3s9V64vL3kuZEyfvgblZ?usp=sharing)


---

## 📋 Overview

This assignment contains **four comprehensive Google Colab notebooks**, each serving as a hands-on tutorial for a major family of deep learning architectures. Together they cover the full spectrum from sequential models to graph neural networks to modern vision transformers.

| # | Notebook | Topic | Cells | Key Models |
|---|----------|-------|-------|------------|
| 1 | [RNN / LSTM / GRU / WaveNet](Final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb) | Sequence modeling from scratch | 70 (34 code) | Vanilla RNN, LSTM, GRU, Deep LSTM, WaveNet |
| 2 | [10 Years of NLP](Final_nlp_deep_learning_10_years_tutorial.ipynb) | NLP evolution 2013 → 2023 | 43 (14 code) | Word2Vec, RNN/LSTM, Attention, Transformer, BERT, GPT |
| 3 | [GNN Fundamentals](Final_gnn_fundamentals_tutorial.ipynb) | Graph neural networks | 38 (18 code) | Message Passing, GCN (NumPy + PyTorch) |
| 4 | [Vision Transformers](Final_vision_transformers_tutorial.ipynb) | Modern computer vision | 29 (20 code) | ViT, CLIP, DINOv2, SAM, Hybrid CNN-Transformer |

---

## 📓 Notebook Summaries

### 1. Sequence Models: Zero to Hero — RNN, LSTM, GRU & WaveNet

**File:** `Final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb`

A ground-up tutorial on sequence modeling using **character-level language modeling** as the running task (generating Shakespeare-like text).

| Part | Title | What You Learn |
|------|-------|----------------|
| 1 | The World of Sequences | Why order matters; sliding-window data pipeline; `CharDataset` |
| 2 | Vanilla RNN | First recurrent net; vanishing gradient intuition; `CharRNN` |
| 3 | LSTM | Gating mechanism (forget, input, output gates); `CharLSTM` |
| 4 | GRU | Simplified gating (reset + update); `CharGRU` |
| 5 | Deep RNNs & Tricks | Stacked layers, dropout, temperature sampling; `DeepCharLSTM` |
| 6 | WaveNet | Causal dilated convolutions for sequences; `CharWaveNet` |
| 7 | Grand Comparison | Loss curves, generation quality, parameter counts |

**Key implementations:** `CharDataset`, `train_model()`, `generate_text()`, `CharRNN`, `CharLSTM`, `CharGRU`, `DeepCharLSTM`, `CausalConv1dBlock`, `CharWaveNet`

---

### 2. 10 Years of Deep Learning in NLP — A Beginner's Journey

**File:** `Final_nlp_deep_learning_10_years_tutorial.ipynb`

A chronological walkthrough of every major NLP milestone from Word2Vec (2013) through ChatGPT/GPT-4 (2023).

| Chapter | Title | What You Learn |
|---------|-------|----------------|
| 1 | Basics of Language Modeling | Tokenization, BPE, word embeddings, Word2Vec analogy tests |
| 2 | Sequential Models | RNNs, Bidirectional RNNs, LSTMs vs GRUs, Attention mechanism |
| 3 | The Transformer | Self-attention, multi-head attention, positional encodings, full architecture |
| 4 | Rise of LLMs | GPT (decoder-only), BERT (encoder-only), XLNet, T5 |
| 5 | Human Alignment | Hallucination, RLHF, InstructGPT → ChatGPT → GPT-4 |

**Highlights:** Interactive 2D embedding visualizations, the famous analogy test ("king − man + woman = queen"), side-by-side LSTM vs GRU comparison tables.

---

### 3. Graph Neural Networks: From Zero to Hero

**File:** `Final_gnn_fundamentals_tutorial.ipynb`

A from-scratch implementation of Graph Neural Networks, progressing from basic graph theory to a full GCN trained on the Karate Club dataset.

| Chapter | Title | What You Learn |
|---------|-------|----------------|
| 1 | What Are Graphs? | Nodes, edges, directed/undirected/weighted graphs |
| 2 | Graph Representations | Adjacency matrix, edge list, node features |
| 3 | Why GNNs? | Permutation invariance problem, limitations of traditional ML |
| 4 | Message Passing | Aggregate → Update paradigm, receptive field growth |
| 5 | GCN from Scratch | Kipf & Welling (2017), normalized adjacency, GCN in NumPy |
| 6 | GCN in PyTorch | `GCNLayerPyTorch`, automatic differentiation |
| 7 | Real Data | Node classification on the Karate Club network |

**Key implementations:** `Graph` class, `simple_message_passing()`, `GCNLayerNumPy`, `GCNNumPy`, `GCNLayerPyTorch`, `karate_club_demo()`

---

### 4. Vision Transformers & The Frontier of Computer Vision

**File:** `Final_vision_transformers_tutorial.ipynb`

Covers the revolution that occurred when Transformers moved from NLP to computer vision, from the original ViT paper through the latest foundation models.

| Chapter | Title | What You Learn |
|---------|-------|----------------|
| 1 | Attention Mechanism | Scaled dot-product attention, query/key/value |
| 2 | Vision Transformer (ViT) | Patch embedding ("an image is 16×16 words"), CLS token |
| 3 | CLIP | Contrastive learning, vision-language pretraining |
| 4 | DINOv2 | Self-supervised visual features, self-distillation |
| 5 | SAM | Segment Anything, prompt-based segmentation |
| 6 | Hybrid Architectures | CNN + Transformer combinations |
| 7 | Practical Applications | Using SOTA models in real projects |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or a local environment with a GPU

### Running in Google Colab

1. Open any notebook in Google Colab
2. Select **Runtime → Change runtime type → T4 GPU**
3. Run all cells sequentially (each notebook is self-contained and installs its own dependencies)

### Dependencies

All notebooks auto-install their dependencies. Core libraries used:

```
torch, numpy, matplotlib, networkx, torch-geometric, 
transformers, scikit-learn, seaborn
```

---


## 📂 Repository Structure

```
RNNs/
├── README.md
├── Final_rnn_lstm_gru_wavenet_zero_to_hero.ipynb   # Notebook 1: Sequence Models
├── Final_nlp_deep_learning_10_years_tutorial.ipynb  # Notebook 2: NLP Evolution
├── Final_gnn_fundamentals_tutorial.ipynb            # Notebook 3: Graph Neural Networks
├── Final_vision_transformers_tutorial.ipynb          # Notebook 4: Vision Transformers
```

---

## 📝 License

This project is for educational purposes as part of CMPE 258 at San José State University.

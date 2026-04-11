# Deep Learning Training Design Space - Tutorial Series

Welcome to the **Deep Learning Training Design Space** repository. This collection of tutorials is designed to provide a comprehensive, hands-on journey through the essential components of modern deep learning, from foundational concepts to state-of-the-art architectures.

## Video Walkthrough
To get a quick overview of the tutorials and their content, check out the video walkthrough: [Deep Learning Training Design Space - Video Walkthrough](https://drive.google.com/drive/folders/1Myf3yVF-n7QauCXBCHfUyG3kahAihuyl?usp=sharing).

## 📚 Tutorial Overview

The repository consists of five core tutorials, each focusing on a critical aspect of the deep learning pipeline:

### 1. [Activation Functions for Deep Learning: From Zero to Hero](./Copy%20of%20final_activation_functions_tutorial.ipynb)
Learn about the mathematical components that allow neural networks to learn complex, non-linear patterns.
*   **Key Topics:** Classic activations (Sigmoid, Tanh), ReLU Revolution (LeakyReLU, ELU, SELU), and Modern Activations (GELU, Swish, Mish).
*   **Focus:** Understanding why activations matter and how to choose the right one for your task.

### 2. [Convolutional Neural Networks from First Principles](./Copy%20of%20final_cnn_fundamentals_tutorial.ipynb)
A visual journey into the architecture that revolutionized computer vision.
*   **Key Topics:** Convolution operations, kernels, feature maps, pooling, and downsampling.
*   **Implementation:** Building an end-to-end CIFAR-10 classifier from scratch.

### 3. [Modern CNN Architectures: From ResNet to EfficientNet](./Copy%20of%20final_modern_cnn_architectures_tutorial.ipynb)
Explore the architectural innovations that enabled training of extremely deep networks.
*   **Key Topics:** ResNet & Skip Connections, the Vanishing Gradient problem, and Transfer Learning.
*   **Advanced Concepts:** Compound scaling with EfficientNet and fine-tuning strategies for real-world applications.

### 4. [Hyperparameter Tuning for Deep Learning: From Zero to Hero](./Copy%20of%20final_hyperparameter_tuning_tutorial.ipynb)
Master the art of optimization to separate "good" models from "great" ones.
*   **Key Topics:** Manual tuning vs. Automated search (Grid, Random, Bayesian).
*   **Tools:** Hands-on tutorial with Optuna for modern hyperparameter optimization.

### 5. [Important Classification Metrics](./Copy%20of%20final_important_classification_metrics_tutorial.ipynb)
Learn how to rigorously evaluate your models beyond simple accuracy.
*   **Key Topics:** Binary, Multi-class, and Multi-label classification definitions and real-world use cases.
*   **Focus:** Understanding the metrics that define success in supervised learning tasks.

## 🛠️ Getting Started

### Prerequisites
To run these notebooks, you will need:
*   Python 3.8+
*   PyTorch / TensorFlow (depending on the specific notebook)
*   Standard DS Stack: `numpy`, `matplotlib`, `scikit-learn`

### Installation
We recommend using a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt  # If applicable
```

## 🚀 Usage
Each notebook is self-contained and designed to be run in **Google Colab** or a local **Jupyter** environment. They include both theoretical explanations and executable code blocks for hands-on learning.

---
*Created as part of the CMPE-258 Deep Learning course assignments.*

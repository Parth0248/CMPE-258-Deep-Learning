# Deep Learning Colabs — 20-Minute Video Summary Script

> Estimated pacing: ~1.5 minutes per notebook. Read each section, briefly show the notebook, and move on.

---

## Part A: Math Foundations (~6 min)

### 1. Linear Algebra for Deep Learning (~1.5 min)

This notebook builds linear algebra intuition from the ground up for deep learning. It starts with **scalars, vectors, matrices, and tensors** — the four data structures you'll see everywhere. Vectors are 1-D arrays (a single data sample), matrices are 2-D (a batch of samples), and tensors are 3-D+ (images, sequences). The notebook then walks through **vector operations** — addition, scalar multiplication, and the **dot product**, which is the single most important operation in deep learning because every neuron computes one. Next, it covers **matrix operations** and the critical **shape rule** for matrix multiplication: `(m×n) @ (n×p) → (m×p)`. This is how batch processing works — one matrix multiply handles an entire layer for an entire batch. It finishes with **special matrix properties** (transpose, inverse, determinant), **eigenvalues/eigenvectors** for understanding transformations, and **norms/distances** for measuring error. The core equation tying it all together: `output = activation(weights @ input + bias)`.

### 2. Calculus for Deep Learning (~1 min)

This notebook covers the calculus that makes neural networks learn. It starts with **derivatives** — the rate of change — and lists the common derivatives you'll encounter (sigmoid, ReLU, tanh). Then it introduces the **chain rule**, which is literally the math behind backpropagation: if `y = f(g(x))`, then `dy/dx = f'(g(x)) · g'(x)`. Next, **partial derivatives and gradients** — when a loss function depends on thousands of weights, the gradient tells us the direction of steepest ascent for each one. **Gradient descent** is the update rule: `w = w - lr × ∂L/∂w`. The notebook concludes with a full **backpropagation** walkthrough — forward pass, compute loss, backward pass via chain rule, update weights — the four-step training loop every deep learning model uses.

### 3. Probability for Deep Learning (~1 min)

A comprehensive 9-part notebook. It covers **probability fundamentals** (sample spaces, axioms), **conditional probability and Bayes' theorem**, **random variables** with PMFs and PDFs, **expected value and variance** (which connect to loss functions), and key **distributions** (Bernoulli for binary classification, Gaussian for weight initialization). The **information theory** section is especially important: **entropy** measures uncertainty, **cross-entropy** is the most common classification loss, and **KL divergence** measures how different two distributions are (used in VAEs). It also covers **maximum likelihood estimation** — training is just maximizing the likelihood of the data — and **sampling methods** including Monte Carlo estimation, mini-batch SGD, and the **reparameterization trick** used in VAEs. Practical exercises implement softmax, cross-entropy loss, and dropout from scratch.

### 4. Probability Fundamentals for Deep Learning (~1 min)

A more gentle, chapter-by-chapter approach to probability. It spans 10 chapters: starting from "What is Probability?" (frequentist vs. Bayesian views), building through the **language of probability** (sample spaces, events), **basic rules** (addition, multiplication), **conditional probability**, and **Bayes' theorem** with sequential updating. Then **random variables**, **probability distributions** (with softmax as a PMF), **expectation and variance**, **MLE**, and **information theory** (entropy, cross-entropy, KL divergence). Chapter 10 ties everything together with a **complete neural network classification pipeline** showing how all these probability concepts combine in practice.

### 5. NumPy Foundations for Deep Learning (~0.5 min)

Covers the NumPy skills needed before touching any framework. Five sections: **array creation** (shapes, dtypes, special arrays), **indexing and slicing** (including batch processing patterns), **element-wise operations** (arithmetic + activation functions like ReLU and sigmoid), **broadcasting** (how NumPy automatically expands shapes — the same rules apply in PyTorch and TensorFlow), and **matrix multiplication** — implementing `output = X @ W + b` which is the dense layer forward pass.

---

## Part B: Neural Networks from Scratch (~3 min)

### 6. Neural Networks from Scratch (Colab 1) (~2 min)

This is the flagship notebook — building a complete neural network using only NumPy. It progresses through 8 chapters. **Chapter 1** models a single neuron: `output = activation(w·x + b)`. **Chapter 2** covers activation functions — sigmoid, tanh, ReLU, and leaky ReLU — explaining why non-linearity is essential (without it, stacking layers is pointless). **Chapter 3** implements loss functions — MSE for regression and cross-entropy for classification. **Chapter 4** builds a full dense layer in matrix form. **Chapter 5** is the heart — **backpropagation** — implementing the chain rule to compute gradients through the entire network. **Chapter 6** assembles the complete network class with forward pass, backward pass, and weight updates. Then the demos: **XOR** (the classic non-linear problem), **moons dataset** (binary classification with curved boundaries), **spirals** (multi-class, highly non-linear), and finally **MNIST digit recognition** — a real-world test. All from scratch, no frameworks.

### 7. Why Deep Learning Works: Geometric Intuition (~1 min)

This notebook answers the question: *why does depth matter?* It starts with the **Universal Approximation Theorem** — a single hidden layer *can* approximate any function, but may need exponentially many neurons. The key insight is **space folding**: each layer with a ReLU activation folds the input space, creating new linear regions. A **shallow network** (1 wide layer) creates regions linearly proportional to width. A **deep network** creates regions **exponentially** proportional to depth. The notebook demonstrates this with the Belgium-Netherlands border problem — a complex, winding boundary. It trains both a shallow and deep network side-by-side and visually shows how the deep network achieves better decision boundaries with far fewer parameters. Takeaway: **depth gives exponential expressiveness**.

---

## Part C: Framework Tutorials (~11 min)

### TensorFlow / Keras Track (~4 min)

#### 8. TensorFlow Tensor Operations Tutorial (~1.5 min)

The TensorFlow counterpart to the NumPy foundations notebook. It covers **tensor creation** (constants, variables, random), **dtypes and attributes**, **NumPy interop**, **indexing/slicing** (including `tf.gather` and boolean masking), **reshaping** (reshape, transpose, expand_dims), **concatenation/stacking**, **element-wise math and activation functions**, **broadcasting** (same rules as NumPy), **linear algebra** (matmul, norms, decompositions), **Einstein summation** (`einsum`) — an incredibly powerful notation for expressing any tensor operation in one line — and **reduction operations** (sum, mean, max across axes). It concludes with common deep learning patterns: implementing a **dense layer**, and **scaled dot-product attention** from scratch using TensorFlow ops.

#### 9. Keras/TensorFlow Neural Networks Tutorial (Colab 3) (~1.5 min)

A complete journey from low-level TensorFlow to high-level Keras. **Part I** reviews TensorFlow tensor fundamentals. **Part II** covers `einsum` for tensor operations. **Part III** introduces **GradientTape** — TensorFlow's automatic differentiation engine — and notes the key difference from PyTorch: you explicitly create a tape context. **Part IV** rebuilds the neural network from scratch using only TensorFlow primitives (`tf.Variable`, `tf.matmul`, `GradientTape`), mirroring the NumPy version from Colab 1. **Part V** introduces the **Keras Sequential and Functional APIs** — showing how the same network becomes just a few lines of code. **Part VI** runs complete training examples. **Part VII** compares the three approaches: NumPy (educational), TensorFlow primitives (flexible), Keras (production). The message: understand the low level, use the high level.

#### 10. Keras/TensorFlow Advanced Tutorial (~1 min)

Picks up where the previous notebook left off. **Advanced GradientTape** patterns (higher-order gradients, persistent tapes). **Building operations from scratch** — implementing individual neural network primitives. **Custom layers using only `tf.Variable`** — no Keras abstractions. Then the proper way: **custom Keras layers** by subclassing `tf.keras.layers.Layer` with `build()` and `call()` methods. **Advanced architectures** — residual connections, attention mechanisms, and modern deep learning components. **Custom training loops** using GradientTape for full control over the training process (useful for GANs, reinforcement learning, etc.). Wraps up with practical demos putting it all together.

### PyTorch Track (~4 min)

#### 11. PyTorch Tensors from Zero to Hero (Colab 4) (~1.5 min)

The most comprehensive tensor tutorial in the set — 10 parts. Covers **tensor creation** (from lists, NumPy, special functions, random), **attributes** (shape, dtype, device — CPU vs GPU), **indexing** (basic, fancy, boolean masking), **operations** (element-wise arithmetic, math functions, aggregations), **broadcasting**, **in-place operations** (the underscore convention: `add_`). **Reshaping** with `view`, `reshape`, `flatten`, `squeeze`/`unsqueeze`, `permute`, and `cat`/`stack`. **Linear algebra** — matmul, dot product, norms. A full **einsum** section covering batch matrix multiply, attention, and a cheatsheet. **Common deep learning operations**: activation functions, softmax, loss functions, normalization, dropout. And **practical patterns**: weight initialization, autograd basics, and common code snippets. This is the PyTorch reference notebook.

#### 12. PyTorch Neural Networks Tutorial (Colab 2) (~1.5 min)

Mirrors the TensorFlow journey but in PyTorch. **Part I**: tensor fundamentals. **Part II**: einsum. **Part III**: **Autograd** — PyTorch's automatic differentiation. Unlike TensorFlow's explicit tape, PyTorch tracks gradients automatically on tensors with `requires_grad=True`. **Part IV**: neural network from scratch using only PyTorch primitives — `torch.randn`, manual matmul, and autograd for backprop. **Part V**: the high-level `nn.Module` API — defining networks by subclassing `nn.Module`, using `nn.Linear`, `nn.ReLU`, optimizers from `torch.optim`, and loss functions from `nn.functional`. **Part VI**: complete training examples on real datasets. **Part VII**: comparison of NumPy vs. PyTorch primitives vs. `nn.Module` — same progression as the TensorFlow track. Key difference emphasized: PyTorch uses **dynamic computation graphs** (define-by-run) vs. TensorFlow's historical static graphs.

#### 13. PyTorch Advanced Tutorial (~1 min)

Advanced PyTorch patterns. **Advanced autograd** — higher-order gradients, custom autograd functions. **Building operations from scratch** — convolution, normalization, etc. implemented manually. **Custom layers using `nn.Parameter`** — the raw way to define learnable parameters. **Custom `nn.Module` layers** — the standard approach with proper `forward()` methods. **Advanced architectures** — residual blocks, attention mechanisms, modern components. **Custom training loops** — bypassing `model.fit()` for full control, useful for complex training schemes. Practical demos tying everything together. Parallels the Keras advanced notebook but in PyTorch idioms.

### JAX Track (~3 min)

#### 14. JAX Neural Networks Tutorial (Colab 4) (~1.5 min)

Introduction to JAX — Google's functional deep learning framework. **Part I**: JAX arrays are almost identical to NumPy, but **immutable** (no in-place operations). **Part II**: JAX's superpowers — **`jit`** (just-in-time compilation for speed), **`vmap`** (automatic vectorization — write code for one sample, apply to a batch), and **`pmap`** (parallel computation across devices). **Part III**: einsum. **Part IV**: **automatic differentiation** with `grad` and `value_and_grad` — JAX can compute Jacobians and Hessians trivially. **Part V**: building a neural network from scratch in a **purely functional** style — no classes, no state, just functions that take parameters and data. **Part VI**: the high-level **Flax Linen** API — `nn.Dense`, `nn.relu`, `@nn.compact` — the production way to write JAX models. **Part VII**: comparison of approaches. Key distinction: JAX is **functional** (pass state explicitly) vs. PyTorch's **stateful** (state lives in objects).

#### 15. JAX Deep Learning Tutorial (Colab 5) (~1.5 min)

A deeper dive into JAX. Covers the **functional programming philosophy** — pure functions, explicit random state via PRNGKeys. **Tensor fundamentals** — array creation, JAX's explicit random number generation (you split keys), dtypes, immutability. **Tensor operations** — element-wise, broadcasting, reductions, reshaping, concatenation. **Linear algebra** for deep learning. This notebook provides more hands-on practice with JAX's unique patterns — especially around immutability and explicit state management — building fluency with the functional paradigm before tackling complex models. It bridges the gap between understanding JAX's basics and using libraries like Flax and Optax for real deep learning projects.

---

## Closing (~30 sec)

To summarize the full set: we started with the **math foundations** — linear algebra, calculus, probability, and NumPy — which are the language of deep learning. Then we **built neural networks from scratch** using only NumPy and understood geometrically *why* depth matters. Finally, we covered three major **frameworks** — TensorFlow/Keras, PyTorch, and JAX — each following the same progression: tensors → autograd → scratch implementation → high-level API → advanced patterns. The key takeaway across all 15 notebooks: **understand the fundamentals first, then use the frameworks**.

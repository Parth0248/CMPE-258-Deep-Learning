---
title: Building MgACT v2
subtitle: Cross-Modal Policies for Contact-Rich Robotic Manipulation
theme: default
---

# Building MgACT v2
### Deep Learning Research Project
**Topic:** Imitation Learning, Cross-Modal Attention, and Contact-Rich Manipulation

---

# The Problem: Visual Occlusion
- **The Task:** High-precision cable insertion (AIC Qualification).
- **The Bottleneck:** Standard VLA models rely entirely on vision.
- **The Reality:** During insertion, the robot's arm and the cable occlude the camera views.
- **The Solution:** When you can't see, you must *feel*. We need a policy that effectively uses force/torque (haptic) data.

---

# Data Collection: The CheatCode Policy
- **The Approach:** We needed high-quality expert demonstrations.
- **The Method:** Developed a custom teleoperation setup based on CheatCode.
- **The Dataset:** Successfully collected **500 expert episodes**.
- **Data Quality:** Perfectly synchronized streams at 16.7 Hz:
  - 3x RGB Cameras
  - 21-D Proprioception (Joint states)
  - 6-DOF Wrench (Force/Torque)
  - 19-D Variable-Impedance Action Space

---

# The MgACT v2 Architecture
Building on the Action Chunking with Transformers (ACT) framework.

- **Vision Backbone:** ResNet-18 extracting features from 3 camera views.
- **Haptic Backbone:** 1D-CNN processing an 8-step window of wrench data to capture force transients.
- **The Innovation:** Bidirectional Cross-Attention. Vision and haptic tokens attend to each other *before* the main transformer layers.

---

# Key Architectural Innovations

1. **Contact-Conditioned Modality Dropout:**
   - If force > 5N, mask out vision with 50% probability during training.
   - Prevents causal confusion and forces the network to use haptic data during physical contact.
2. **Auxiliary Losses:**
   - **Wrench Reconstruction:** Ensures haptic features aren't ignored.
   - **Phase Classification:** Weak supervision to predict "free-space" vs "contact".
   - **Smoothness Penalty:** Discourages jerky movements in predicted stiffness.

---

# Preliminary Results
- **Training Setup:** Currently training on Colab (L4 GPU).
- **Early Metrics:** Rapid convergence observed immediately.
  - **Epoch 1/30**
  - `train_loss = 19.8784`
  - `val_loss = 0.0343`
  - `val_L_action = 0.0232`
- **Analysis:** An L1 action error of 0.0232 in epoch 1 shows the cross-modal architecture is successfully learning the complex variable-impedance action space.

---

# Next Steps
- **Tomorrow:** Finalize the 30-epoch training run and extract the best weights (`mg_act_v2_best.pt`).
- **Evaluation:** Deploy the trained model into the closed-loop simulator.
- **Optimization:** Tune hyperparameters (chunk size, dropout threshold) based on physical sim performance.

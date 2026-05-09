# Deep Learning Assignment: MgACT - A Cross-Modal Policy for Robotic Insertion

## Overview
This repository contains the deliverables for our Deep Learning research project, focusing on the development and training of **Modality Gated Cross-Attention Transformers (MgACT)**. 

Moving beyond standard vision-language-action models, we are tackling a contact-rich manipulation task (cable insertion for the AIC Qualification) by implementing a novel cross-modal architecture. Our model fuses synchronized vision, proprioception, and high-frequency haptic (wrench) data to perform precise insertions even under severe visual occlusion.

## Project Progress & Current Status

We have successfully completed the data collection phase and are currently training the model. 
- **Data Collection:** Collected **500 high-quality expert demonstration episodes** using a custom teleoperation policy based on CheatCode. The data features perfectly synchronized vision, joint state, and 6-DOF force/torque logs at 16.7 Hz.
- **Architecture Setup:** Implemented the MgACT architecture, featuring bidirectional cross-attention between haptic and visual tokens, contact-conditioned modality dropout, and auxiliary losses (wrench reconstruction and phase classification).
- **Training Status:** Training is currently ongoing on a Colab L4 GPU.
- **Performance Analysis:** The model is learning effectively, with the training loss decreasing steadily and the validation loss dropping significantly in the first few epochs. The auxiliary losses are also converging, indicating that the model is learning to reconstruct the input data and predict the phase correctly. The action loss is the main focus, and we are monitoring it closely to ensure that the model is learning to output the correct actions. 

## Deliverables

1. **Strategy & Architecture Plan** (`MG_ACT_v2_Strategy.md`)
   - The detailed blueprint defining our approach, the neural network structure, and our 7-day execution plan.

2. **Training Notebook** (`train_mg_act_v2.ipynb`)
   - The Colab notebook used to train the MgACT v2 policy.

5. **Model Weights** (`mg_act_v2_best.pt`)
   - The best checkpoint from our training run (to be finalized tomorrow).

## Medium Article
[From CheatCode to Cross-Modal Attention: Building MgACT for Contact-Rich Manipulation](https://medium.com/@parthmaradia2002/from-cheatcode-to-cross-modal-attention-building-mgact-for-contact-rich-manipulation-e3ffdc4d7a4a)

## DATA SET AND CHECKPOINTS
[DATASET AND CHECKPOINTS](https://drive.google.com/drive/folders/15eKg8txeow-CUvSdWRcAJp4hBBCGaUCo?usp=sharing)

## VIDEO PRESENTATION
[Video Walkthrough](https://drive.google.com/drive/folders/1S924bz5uboDPCXV--rfOmjkMBvwWO366?usp=sharing)

## Slides
[SLIDES](https://docs.google.com/presentation/d/1H2IRfr87PwfkLHKHqqXVxX7lhI-HNKYNH8c8XfSi2hw/edit?usp=sharing)

## Training Colab
[Colab Notebook](https://colab.research.google.com/drive/1ZjLUndn38EbxCt_Cm8QCe3jcC6o2SEg2?usp=sharing)

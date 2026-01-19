# 🌼 FlowerDiff: Diffusion Models & Embedding Analysis

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/Consscht/FlowerDiff)
[![Weights & Biases](https://img.shields.io/badge/W%26B-Experiment%20Logs-orange)]([https://wandb.ai/](https://wandb.ai/Conscht-Sht/Hands-on-CV-Project3))

This repository contains the implementation for the **Denoising Probabilistic Diffusion Models** student assessment. The project focuses on generating high-quality images using a U-Net DDPM, extracting internal representations ("embeddings"), and evaluating the results using modern MLOps tools.

## 🎯 Project Goals

1.  **Generate** flower images from text prompts (e.g., "A red rose") using a pre-trained U-Net.
2.  **Extract** intermediate embeddings from the U-Net's bottleneck (`down2` layer) using PyTorch hooks.
3.  **Evaluate** quality using **CLIP Score** (semantic alignment) and **FID** (distribution distance).
4.  **Analyze** the dataset using **FiftyOne** to find unique and representative samples.
5.  **Bonus:** Build an MNIST classifier with an "I Don't Know" (IDK) option for ambiguous generations.

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Conscht/DiffFlower.git](https://github.com/Conscht/DiffFlower.git)
   cd DiffFlower

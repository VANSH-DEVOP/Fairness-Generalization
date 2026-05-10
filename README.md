# Preserving Fairness Generalization in Deepfake Detection

## Overview

This project is based on the CVPR 2024 research paper:

**“Preserving Fairness Generalization in Deepfake Detection”**

The objective of the paper is to improve fairness in deepfake detection systems by reducing demographic bias while maintaining strong detection performance.

Traditional deepfake detectors often become biased toward dominant demographic groups present in training datasets. This project studies and reproduces the fairness-aware deepfake detection framework proposed in the paper.

---

# Research Objective

The main goal of this project is to:

- Understand fairness-aware deepfake detection
- Reproduce the research implementation
- Execute the inference pipeline successfully
- Adapt the repository for Apple Silicon compatibility
- Study feature disentanglement and fairness generalization

---

# Key Concepts

The model focuses on:

- Deepfake Detection
- Fairness Generalization
- Feature Disentanglement
- Demographic Bias Reduction
- Multi-Encoder Representation Learning

---

# Architecture Summary

The proposed framework uses multiple Xception-based encoders to separately learn:

| Encoder | Purpose |
|---|---|
| Encoder F | Forgery feature extraction |
| Encoder C | Facial/content feature extraction |
| Encoder Fair | Fairness/demographic feature extraction |

The architecture also includes:

- Conditional GAN modules
- AdaIN-based feature manipulation
- Multiple classification heads
- Fairness-aware loss functions
- Fused prediction mechanism

---

# Why Xception?

Xception is used as the backbone architecture because it efficiently captures:

- Fine-grained textures
- Manipulation artifacts
- Compression inconsistencies
- Facial forgery patterns

The model uses depthwise separable convolutions for efficient feature extraction.

---

# Processing Pipeline

The inference pipeline follows these stages:

1. Image Loading
2. Image Preprocessing
3. Feature Extraction using Xception Encoders
4. Feature Disentanglement
5. Fairness-aware Representation Learning
6. Forward Propagation
7. Fused Classification
8. Prediction Generation

---

# What Was Implemented in This Project

The following tasks were successfully completed:

- Repository setup and configuration
- PyTorch environment setup
- Apple Silicon compatibility adaptation
- CUDA-specific code patching
- Device-agnostic tensor handling
- Checkpoint loading stabilization
- Dataset loading fixes
- Prediction output improvements
- Custom demo dataset creation
- End-to-end inference pipeline execution

---

# Apple Silicon Compatibility Fixes

The original repository was designed primarily for NVIDIA CUDA systems.

The following modifications were made to successfully execute the project on Apple Silicon (M4):

- Removed CUDA-only tensor assumptions
- Patched checkpoint loading with device-aware mapping
- Added CPU/MPS-compatible tensor handling
- Fixed file saving path issues
- Stabilized inference execution
- Added readable prediction logs

---

# Example Inference Output

The model successfully generated prediction outputs including:

- Image path
- Ground truth label
- Prediction confidence
- Predicted class

Example:

```bash
Image: ../demo_data/images/fake_0.jpg
Ground Truth: Fake
Fake Confidence: -0.0277
Predicted Class: Real
```

---

# Technologies Used

- Python
- PyTorch
- torchvision
- OpenCV
- albumentations
- timm
- segmentation-models-pytorch

---

# Repository Structure

```bash
training/
│
├── detectors/
├── dataset/
├── networks/
├── loss/
├── pretrained/
├── train.py
├── test.py
```

---

# Running the Inference Pipeline

```bash
python test.py \
--test_path ../demo_data/test.csv \
--savepath ../demo_results \
--batch_size 1 \
--inter_attribute male,white \
--single_attribute male-nonmale
```

---

# Challenges Faced

- CUDA-specific implementation assumptions
- Apple Silicon compatibility issues
- Missing trained fairness detector checkpoints
- Dataset subgroup filtering problems
- File path handling issues

---

# Learning Outcomes

This project helped in understanding:

- Deepfake forensic analysis
- Fairness-aware AI systems
- Xception architecture
- PyTorch-based research implementations
- Feature disentanglement
- Inference pipeline execution
- Research repository debugging and stabilization

---

# Future Improvements

Possible improvements include:

- Real-time webcam deepfake detection
- Transformer-based backbones
- Explainable AI visualization (Grad-CAM)
- Multimodal detection using audio + video
- Lightweight deployment for edge devices
- Better fairness-aware optimization strategies

---

# Important Note

The repository provides pretrained Xception backbone weights but does not include the final trained fairness detector checkpoint.

The current implementation successfully executes the complete inference pipeline using pretrained backbone initialization.

---

# References

1. Li Lin, Xinan He, Yan Ju, Xin Wang, Feng Ding, Shu Hu  
   **“Preserving Fairness Generalization in Deepfake Detection”**  
   CVPR 2024

2. François Chollet  
   **“Xception: Deep Learning with Depthwise Separable Convolutions”**  
   CVPR 2017

3. Official Repository:  
   https://github.com/Purdue-M2/Fairness-Generalization

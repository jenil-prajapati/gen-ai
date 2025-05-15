# YOLO Object Detection Implementation

## Overview
This project implements YOLOv1 (You Only Look Once) object detection model using PyTorch, based on the Redmon et al. paper. The model performs real-time object detection by framing it as a single regression problem, predicting bounding boxes and class probabilities directly from full images.

## Model Architecture
- **Backbone**: ResNet-34 pretrained on ImageNet
- **Detection Head**: 4 Convolutional layers followed by 2 Fully Connected layers
- **Output**: (S × S × (5B + C)) tensor (7 × 7 × 30)
- **Loss**: Custom YOLO loss combining localization, objectness, and classification terms

## Dataset
- **Dataset Used**: PASCAL VOC 2012
- **Custom Split**: 80%-20% (train_custom.txt, val_custom.txt)
- **Training Set**: 9232 images
- **Validation Set**: 2308 images

## Performance
- Achieved mean Average Precision (mAP) of 9.11% on validation set
- Evaluated at IoU 0.5 and confidence threshold 0.3
- NMS (Non-Maximum Suppression) applied to filter duplicate detections
- Best training loss ≈ 4.98 around epoch 20

## Training Details
- **Epochs**: 20
- **Batch Size**: 8
- **Optimizer**: SGD
  - Learning Rate: 1e-4
  - Momentum: 0.9
  - Weight Decay: 5e-4
- Learning rate step decay at epoch 1

## Run Instructions

### Environment Setup
Set up a Python 3 environment with the following required packages:
- PyTorch
- torchvision
- albumentations
- OpenCV
- matplotlib
- tqdm

### Dataset Preparation
1. Upload the code files
2. Prepare VOC 2012 dataset with custom splits:
   - train_custom.txt
   - val_custom.txt

### Running the Pipeline
The project provides two main functions:
- `train(config)`: To train the model
- `infer(config)`: To visualize sample detections

GPU acceleration is automatically utilized if available.

## Key Learnings
- Built a complete object detection pipeline from scratch
- Implemented custom loss function and model architecture
- Gained experience in training/debugging deep learning models
- Learned to handle challenges like:
  - Convolution layering effects
  - Slow convergence
  - Low and high confidence predictions
  - Complex training and inference issuesx

This implementation provides practical experience in working with large datasets, training custom models, and validating results under real-world conditions.

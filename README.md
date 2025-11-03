# FairFace Multi-Task Classification

A deep learning model that simultaneously predicts age, gender, and race from facial images.

## Results

After 10 epochs of training:
- Gender Accuracy: 70.6%
- Race Accuracy: 40.5%
- Age MAE: 0.144

## Model Architecture

- Backbone: VGG16 with custom multi-task heads
- Tasks: Age regression, gender classification, race classification
- Loss: Weighted combination of L1, BCE, and CrossEntropy losses

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt

2. Run training:
python main.py

Project Structure
models/ - Model architecture

training/ - Training scripts

utils/ - Helper functions

main.py - Main training script

Future Improvements
Try ResNet backbones

Implement UNet architecture

Improve race classification accuracy

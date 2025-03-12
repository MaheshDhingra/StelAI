# StelAI 1

StelAI 1 is an open-source, multi-modal AI system built with PyTorch. It is designed to handle text, images, audio, and video inputs and incorporates a Mixture-of-Experts (MoE) layer for efficient computation. The project includes placeholder functions for internet search and RLHF (Reinforcement Learning with Human Feedback) updates.

## Repository Structure

- **model.py**: Contains the definition of the StelAI1 model along with modality-specific encoders and an MoE layer.
- **tokenizer.py**: A simple tokenizer for processing text data.
- **train.py**: A training script that demonstrates how to use the model and tokenizer with dummy data.
- **utils.py**: Utility functions for saving and loading configurations.
- **requirements.txt**: List of Python packages required for the project.

## Getting Started

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt

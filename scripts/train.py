"""
Train CNN3 for Brain MRI binary classification.

This script uses the reusable project modules:
- models/cnn3.py
- utils/dataset.py
- utils/trainer.py
"""

from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn, optim


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from models.cnn3 import CNN3
from utils.dataset import get_data_loaders
from utils.trainer import get_default_device, train_model


RANDOM_SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
MODEL_SAVE_PATH = "results/models/cnn3_best.pth"


def set_random_seed(seed):
    """Set random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_configuration(device):
    """Print the training configuration before training starts."""
    print("=" * 50)
    print("Brain MRI Classification Training")
    print("=" * 50)
    print("Model: CNN3")
    print(f"Device: {device}")
    print("Dataset: data/split/")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("Optimizer: Adam")
    print("Loss Function: CrossEntropyLoss")
    print("=" * 50)


def main():
    """Run CNN3 training."""
    try:
        set_random_seed(RANDOM_SEED)

        device = get_default_device()
        print_configuration(device)

        train_loader, val_loader, _test_loader, class_names = get_data_loaders(
            batch_size=BATCH_SIZE
        )

        print(f"Classes: {class_names}")

        model = CNN3()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=EPOCHS,
            save_path=MODEL_SAVE_PATH,
        )

        print("\nTraining finished successfully.")

    except Exception as error:
        raise RuntimeError(f"Training script failed: {error}") from error


if __name__ == "__main__":
    main()

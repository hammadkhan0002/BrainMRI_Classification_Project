"""
CNN5 model for Brain MRI binary classification.

Input shape:
    Grayscale MRI image tensor with shape (batch_size, 1, 224, 224)

Output:
    Logits for 2 classes.
"""

import torch
from torch import nn


class CNN5(nn.Module):
    """
    Five-block convolutional neural network for binary MRI classification.

    The model uses a feature extractor followed by a fully connected classifier.
    """

    def __init__(self, num_classes=2):
        """Initialize CNN5 layers."""
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 x 224 x 224 -> 32 x 112 x 112
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 2: 32 x 112 x 112 -> 64 x 56 x 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 3: 64 x 56 x 56 -> 128 x 28 x 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 4: 128 x 28 x 28 -> 256 x 14 x 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 5: 256 x 14 x 14 -> 512 x 7 x 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """Run a forward pass through the feature extractor and classifier."""
        x = self.features(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    model = CNN5()
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model(dummy_input)

    print("Model summary:")
    print(model)
    print(f"Output shape: {output.shape}")

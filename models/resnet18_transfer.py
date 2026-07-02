"""
Transfer-learning ResNet18 model for Brain MRI binary classification.

Input shape:
    Grayscale MRI image tensor with shape (batch_size, 1, 224, 224)

Output:
    Logits for 2 classes.
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Transfer(nn.Module):
    """
    Pretrained ResNet18 adapted for grayscale MRI binary classification.

    The first convolution is changed to accept one-channel images, and only
    layer4 plus the final classifier are left trainable.
    """

    def __init__(self, num_classes=2):
        """Initialize the transfer-learning ResNet18 model."""
        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        original_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        with torch.no_grad():
            self.model.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all parameters except layer4 and the final classifier."""
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        for parameter in self.model.layer4.parameters():
            parameter.requires_grad = True

        for parameter in self.model.fc.parameters():
            parameter.requires_grad = True

    def forward(self, x):
        """Run a forward pass through the adapted ResNet18 model."""
        return self.model(x)


if __name__ == "__main__":
    model = ResNet18Transfer()
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model(dummy_input)

    print("Model summary:")
    print(model)
    print(f"Output shape: {output.shape}")

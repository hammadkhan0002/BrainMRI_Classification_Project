"""
Transfer-learning SqueezeNet 1.1 model for Brain MRI binary classification.

Input shape:
    Grayscale MRI image tensor with shape (batch_size, 1, 224, 224)

Output:
    Logits for 2 classes.
"""

import torch
from torch import nn
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1


class SqueezeNetTransfer(nn.Module):
    """
    Pretrained SqueezeNet 1.1 adapted for grayscale MRI binary classification.

    The first convolution is changed to accept one-channel images, and the
    classifier output convolution is changed to produce two class logits.
    """

    def __init__(self, num_classes=2):
        """Initialize the transfer-learning SqueezeNet 1.1 model."""
        super().__init__()

        self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)

        original_conv = self.model.features[0]
        self.model.features[0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        with torch.no_grad():
            self.model.features[0].weight.copy_(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

            if original_conv.bias is not None:
                self.model.features[0].bias.copy_(original_conv.bias)

        # Keep classifier dropout and activation/pooling unchanged.
        self.model.classifier[1] = nn.Conv2d(
            in_channels=512,
            out_channels=num_classes,
            kernel_size=1,
        )
        self.model.num_classes = num_classes

        self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all parameters except features[-1] and the classifier."""
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        for parameter in self.model.features[-1].parameters():
            parameter.requires_grad = True

        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = True

    def forward(self, x):
        """Run a forward pass through the adapted SqueezeNet model."""
        return self.model(x)


if __name__ == "__main__":
    model = SqueezeNetTransfer()
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model(dummy_input)

    print("Model summary:")
    print(model)
    print(f"Output shape: {output.shape}")

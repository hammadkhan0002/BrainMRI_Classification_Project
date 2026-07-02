"""
Transfer-learning MobileNetV2 model for Brain MRI binary classification.

Input shape:
    Grayscale MRI image tensor with shape (batch_size, 1, 224, 224)

Output:
    Logits for 2 classes.
"""

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class MobileNetV2Transfer(nn.Module):
    """
    Pretrained MobileNetV2 adapted for grayscale MRI binary classification.

    The first convolution is changed to accept one-channel images, and only
    the final feature block plus classifier are left trainable.
    """

    def __init__(self, num_classes=2):
        """Initialize the transfer-learning MobileNetV2 model."""
        super().__init__()

        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        original_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        with torch.no_grad():
            self.model.features[0][0].weight.copy_(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

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
        """Run a forward pass through the adapted MobileNetV2 model."""
        return self.model(x)


if __name__ == "__main__":
    model = MobileNetV2Transfer()
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model(dummy_input)

    print("Model summary:")
    print(model)
    print(f"Output shape: {output.shape}")

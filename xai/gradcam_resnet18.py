"""
Generate Grad-CAM visualizations for ResNet18Transfer Brain MRI classification.

Usage:
    python xai/gradcam_resnet18.py path/to/image.png

Outputs are saved to:
    results/gradcam/resnet18/
"""

from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from models.resnet18_transfer import ResNet18Transfer
from utils.trainer import get_default_device


CHECKPOINT_PATH = Path("results/models/resnet18_best.pth")
OUTPUT_DIR = Path("results/gradcam/resnet18")
IMAGE_SIZE = (224, 224)

GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]


class GradCAM:
    """Grad-CAM helper for capturing activations and gradients."""

    def __init__(self, model, target_layer):
        """Register hooks on the target convolutional layer."""
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, _module, _input, output):
        """Save forward activations."""
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output):
        """Save backward gradients."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_index=None):
        """Generate a Grad-CAM heatmap for one input tensor."""
        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_index is None:
            class_index = output.argmax(dim=1).item()

        score = output[:, class_index]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()

        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, class_index

    def close(self):
        """Remove registered hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def load_image(image_path):
    """Load a grayscale image and return original and transformed versions."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_image = Image.open(image_path).convert("L").resize(IMAGE_SIZE)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD),
        ]
    )

    input_tensor = transform(original_image).unsqueeze(0)

    return original_image, input_tensor


def load_model(device):
    """Load ResNet18Transfer checkpoint."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state_dict'")

    model = ResNet18Transfer()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def create_heatmap_image(cam):
    """Convert a normalized Grad-CAM array into a red heatmap image."""
    cam_image = Image.fromarray(np.uint8(cam * 255)).resize(IMAGE_SIZE)

    heatmap = Image.new("RGBA", IMAGE_SIZE)

    for y in range(IMAGE_SIZE[1]):
        for x in range(IMAGE_SIZE[0]):
            intensity = cam_image.getpixel((x, y))
            heatmap.putpixel((x, y), (255, 0, 0, intensity))

    return heatmap


def create_overlay(original_image, heatmap):
    """Overlay the Grad-CAM heatmap on the original grayscale image."""
    original_rgba = original_image.convert("RGBA")
    return Image.alpha_composite(original_rgba, heatmap)


def save_outputs(original_image, heatmap, overlay, image_path, class_index):
    """Save original, heatmap, and overlay images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem

    original_path = OUTPUT_DIR / f"{stem}_original.png"
    heatmap_path = OUTPUT_DIR / f"{stem}_heatmap_class_{class_index}.png"
    overlay_path = OUTPUT_DIR / f"{stem}_overlay_class_{class_index}.png"

    original_image.save(original_path)
    heatmap.save(heatmap_path)
    overlay.save(overlay_path)

    return original_path, heatmap_path, overlay_path


def main(image_path):
    """Run Grad-CAM for a single MRI image."""
    try:
        device = get_default_device()
        print(f"Using device: {device}")

        original_image, input_tensor = load_image(image_path)
        input_tensor = input_tensor.to(device)

        model = load_model(device)

        gradcam = GradCAM(model=model, target_layer=model.model.layer4)

        cam, class_index = gradcam.generate(input_tensor)
        gradcam.close()

        heatmap = create_heatmap_image(cam)
        overlay = create_overlay(original_image, heatmap)

        original_path, heatmap_path, overlay_path = save_outputs(
            original_image=original_image,
            heatmap=heatmap,
            overlay=overlay,
            image_path=image_path,
            class_index=class_index,
        )

        print("Grad-CAM completed successfully.")
        print(f"Predicted class index: {class_index}")
        print(f"Original image saved to: {original_path}")
        print(f"Heatmap image saved to: {heatmap_path}")
        print(f"Overlay image saved to: {overlay_path}")

    except Exception as error:
        raise RuntimeError(f"Grad-CAM generation failed: {error}") from error


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage example:")
        print("python xai/gradcam_resnet18.py data/split/test/tumor/example.png")
        sys.exit(1)

    main(sys.argv[1])

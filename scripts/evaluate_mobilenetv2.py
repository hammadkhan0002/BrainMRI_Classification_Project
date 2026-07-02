"""
Evaluate the saved MobileNetV2Transfer model on the Brain MRI test dataset.

This script loads the best MobileNetV2Transfer checkpoint and computes standard
classification metrics on the test split.
"""

from pathlib import Path
import sys

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from models.mobilenetv2_transfer import MobileNetV2Transfer
from utils.dataset import get_data_loaders
from utils.trainer import get_default_device


CHECKPOINT_PATH = Path("results/models/mobilenetv2_best.pth")
EVALUATION_OUTPUT_PATH = Path("results/evaluation/mobilenetv2_evaluation.txt")
BATCH_SIZE = 32


def load_checkpoint(checkpoint_path, device):
    """Load a saved PyTorch checkpoint with error handling."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        return torch.load(checkpoint_path, map_location=device)
    except Exception as error:
        raise RuntimeError(f"Failed to load checkpoint: {error}") from error


def load_model(checkpoint, device):
    """Create MobileNetV2Transfer and load the saved model weights."""
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state_dict'")

    model = MobileNetV2Transfer()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def run_inference(model, test_loader, device):
    """Run model inference on the test set and collect predictions."""
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if not all_labels:
        raise ValueError("No test labels were collected during inference")

    return all_labels, all_predictions


def compute_metrics(true_labels, predictions, class_names):
    """Compute classification metrics for the test predictions."""
    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(
            true_labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            true_labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            true_labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(true_labels, predictions),
        "classification_report": classification_report(
            true_labels,
            predictions,
            target_names=class_names,
            zero_division=0,
        ),
    }


def format_metrics(metrics, class_names):
    """Format metrics for console output and saving to a text file."""
    lines = [
        "MobileNetV2Transfer Test Evaluation",
        "=" * 20,
        f"Classes: {class_names}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1-score: {metrics['f1_score']:.4f}",
        "",
        "Confusion Matrix:",
        str(metrics["confusion_matrix"]),
        "",
        "Classification Report:",
        metrics["classification_report"],
    ]

    return "\n".join(lines) + "\n"


def save_metrics(report_text, output_path):
    """Save evaluation metrics to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


def main():
    """Evaluate the saved MobileNetV2Transfer model on the test dataset."""
    try:
        device = get_default_device()
        print(f"Using device: {device}")

        _train_loader, _val_loader, test_loader, class_names = get_data_loaders(
            batch_size=BATCH_SIZE
        )

        checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
        model = load_model(checkpoint, device)

        true_labels, predictions = run_inference(model, test_loader, device)
        metrics = compute_metrics(true_labels, predictions, class_names)
        report_text = format_metrics(metrics, class_names)

        print(report_text)
        save_metrics(report_text, EVALUATION_OUTPUT_PATH)

        print("Evaluation completed successfully.")
        print(f"Metrics saved to: {EVALUATION_OUTPUT_PATH}")

    except Exception as error:
        raise RuntimeError(f"Evaluation failed: {error}") from error


if __name__ == "__main__":
    main()

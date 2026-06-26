"""
Reusable PyTorch training engine for Brain MRI classification models.

This module trains any compatible PyTorch classification model, including:
CNN3, CNN4, CNN5, ResNet18, MobileNet, and SqueezeNet.

It does not define model architectures, dataset loaders, or evaluation metrics
beyond training and validation accuracy.
"""

from pathlib import Path

import torch


EARLY_STOPPING_PATIENCE = 5


def get_default_device():
    """Return the best available device: Apple Silicon MPS, CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def validate_training_inputs(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    save_path,
):
    """Validate required training inputs before starting training."""
    if model is None:
        raise ValueError("model cannot be None")

    if train_loader is None:
        raise ValueError("train_loader cannot be None")

    if val_loader is None:
        raise ValueError("val_loader cannot be None")

    if criterion is None:
        raise ValueError("criterion cannot be None")

    if optimizer is None:
        raise ValueError("optimizer cannot be None")

    if device is None:
        raise ValueError("device cannot be None")

    if epochs <= 0:
        raise ValueError(f"epochs must be greater than 0. Received: {epochs}")

    if not save_path:
        raise ValueError("save_path must be provided")


def initialize_history():
    """Create the history dictionary used for plotting later."""
    return {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }


def calculate_accuracy(outputs, labels):
    """Calculate batch accuracy from model outputs and labels."""
    _, predictions = torch.max(outputs, dim=1)
    correct_predictions = (predictions == labels).sum().item()

    return correct_predictions


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch and return loss and accuracy."""
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += calculate_accuracy(outputs, labels)
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("train_loader produced zero samples")

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_correct / total_samples

    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, val_loader, criterion, device):
    """Validate the model for one epoch and return loss and accuracy."""
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += calculate_accuracy(outputs, labels)
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("val_loader produced zero samples")

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_correct / total_samples

    return epoch_loss, epoch_accuracy


def save_best_model(model, optimizer, best_val_accuracy, epoch, save_path):
    """Save the current best model checkpoint."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_accuracy": best_val_accuracy,
        "epoch": epoch,
    }

    torch.save(checkpoint, save_path)


def print_epoch_summary(
    epoch,
    epochs,
    train_loss,
    train_accuracy,
    val_loss,
    val_accuracy,
    best_val_accuracy,
):
    """Print a readable summary for one epoch."""
    print(f"\nEpoch {epoch}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    save_path,
):
    """
    Train a PyTorch classification model with validation and early stopping.

    Returns:
        history: dictionary containing train/validation loss and accuracy lists
    """
    validate_training_inputs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        save_path=save_path,
    )

    model = model.to(device)
    history = initialize_history()

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        try:
            train_loss, train_accuracy = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            val_loss, val_accuracy = validate_one_epoch(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
            )
        except Exception as error:
            raise RuntimeError(f"Training failed at epoch {epoch}: {error}") from error

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_best_model(
                model=model,
                optimizer=optimizer,
                best_val_accuracy=best_val_accuracy,
                epoch=epoch,
                save_path=save_path,
            )
        else:
            epochs_without_improvement += 1

        print_epoch_summary(
            epoch=epoch,
            epochs=epochs,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            best_val_accuracy=best_val_accuracy,
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "\nEarly stopping triggered. "
                f"Validation accuracy did not improve for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

    print("\nTraining completed successfully.")
    print("Best model saved to:")
    print(save_path)

    return history

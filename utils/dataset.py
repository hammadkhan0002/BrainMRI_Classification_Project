"""
Reusable PyTorch dataset loaders for Brain MRI classification.

Expected folder structure:

data/split/
    train/
        tumor/
        no_tumor/
    val/
        tumor/
        no_tumor/
    test/
        tumor/
        no_tumor/
"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_DIR = Path("data/split")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["no_tumor", "tumor"]

GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]


def validate_split_structure(data_dir):
    """Validate that the expected split and class folders exist."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset split directory not found: {data_dir}")

    for split_name in ["train", "val", "test"]:
        split_dir = data_dir / split_name

        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")

            image_files = list(class_dir.glob("*.png"))

            if not image_files:
                raise ValueError(f"No PNG images found in: {class_dir}")


def get_train_transforms():
    """Return transforms used for training images."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD),
        ]
    )


def get_eval_transforms():
    """Return transforms used for validation and testing images."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD),
        ]
    )


def create_image_folder(split_dir, transform):
    """Create an ImageFolder dataset with error handling."""
    try:
        return datasets.ImageFolder(root=split_dir, transform=transform)
    except Exception as error:
        raise RuntimeError(f"Failed to create dataset from {split_dir}: {error}") from error


def create_data_loader(dataset, batch_size, shuffle, num_workers):
    """Create a DataLoader for a dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def get_data_loaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=2,
):
    """
    Create train, validation, and test DataLoaders.

    Returns:
        train_loader
        val_loader
        test_loader
        class_names
    """
    data_dir = Path(data_dir)
    validate_split_structure(data_dir)

    train_dataset = create_image_folder(
        data_dir / "train",
        transform=get_train_transforms(),
    )
    val_dataset = create_image_folder(
        data_dir / "val",
        transform=get_eval_transforms(),
    )
    test_dataset = create_image_folder(
        data_dir / "test",
        transform=get_eval_transforms(),
    )

    train_loader = create_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = create_data_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = create_data_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    class_names = train_dataset.classes

    if class_names != val_dataset.classes or class_names != test_dataset.classes:
        raise ValueError(
            "Class names do not match across train, validation, and test datasets."
        )

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    loaders = get_data_loaders()
    train_loader, val_loader, test_loader, class_names = loaders

    print("DataLoaders created successfully.")
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

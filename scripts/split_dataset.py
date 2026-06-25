"""
Split processed BraTS MRI PNG images into train, validation, and test sets.

The split is performed independently for each class to preserve class balance.
Images are copied, not moved. Existing split output is cleared before each run.
"""

from pathlib import Path
import random
import shutil


RANDOM_SEED = 42

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/split")
REPORT_PATH = OUTPUT_DIR / "split_report.txt"

CLASSES = ["tumor", "no_tumor"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".png"}


def validate_ratios():
    """Ensure the split ratios add up to 1.0."""
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    if round(total_ratio, 5) != 1.0:
        raise ValueError(
            "Split ratios must add up to 1.0. "
            f"Current total: {total_ratio}"
        )


def clear_existing_split():
    """Remove an existing split directory so each run starts clean."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def create_output_dirs():
    """Create train, validation, and test class folders."""
    for split_name in ["train", "val", "test"]:
        for class_name in CLASSES:
            output_path = OUTPUT_DIR / split_name / class_name
            output_path.mkdir(parents=True, exist_ok=True)


def get_class_images(class_name):
    """Return sorted PNG images for one class."""
    class_dir = INPUT_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Missing input class folder: {class_dir}")

    images = sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise ValueError(f"No PNG images found in: {class_dir}")

    return images


def split_images(images):
    """Shuffle and split images into train, validation, and test lists."""
    shuffled_images = images.copy()
    random.shuffle(shuffled_images)

    total_images = len(shuffled_images)
    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)

    train_images = shuffled_images[:train_count]
    val_images = shuffled_images[train_count : train_count + val_count]
    test_images = shuffled_images[train_count + val_count :]

    return train_images, val_images, test_images


def copy_images(images, split_name, class_name):
    """Copy images into the requested split and class folder."""
    destination_dir = OUTPUT_DIR / split_name / class_name

    for image_path in images:
        destination_path = destination_dir / image_path.name
        shutil.copy2(image_path, destination_path)


def write_report(stats):
    """Write a report describing the generated dataset split."""
    total_images = stats["tumor"]["total"] + stats["no_tumor"]["total"]
    train_images = stats["tumor"]["train"] + stats["no_tumor"]["train"]
    val_images = stats["tumor"]["val"] + stats["no_tumor"]["val"]
    test_images = stats["tumor"]["test"] + stats["no_tumor"]["test"]

    report_lines = [
        "BraTS MRI Dataset Split Report",
        "=" * 31,
        f"Total images: {total_images}",
        f"Train images: {train_images}",
        f"Validation images: {val_images}",
        f"Test images: {test_images}",
        f"Tumor images: {stats['tumor']['total']}",
        f"No-tumor images: {stats['no_tumor']['total']}",
        f"Split ratio used: {int(TRAIN_RATIO * 100)}% train, "
        f"{int(VAL_RATIO * 100)}% validation, "
        f"{int(TEST_RATIO * 100)}% test",
        f"Random seed: {RANDOM_SEED}",
        "",
        "Class breakdown:",
    ]

    for class_name in CLASSES:
        report_lines.extend(
            [
                f"{class_name}:",
                f"  Total: {stats[class_name]['total']}",
                f"  Train: {stats[class_name]['train']}",
                f"  Validation: {stats[class_name]['val']}",
                f"  Test: {stats[class_name]['test']}",
            ]
        )

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main():
    """Create the train, validation, and test split."""
    validate_ratios()
    random.seed(RANDOM_SEED)

    clear_existing_split()
    create_output_dirs()

    stats = {}

    for class_name in CLASSES:
        try:
            images = get_class_images(class_name)
            train_images, val_images, test_images = split_images(images)

            copy_images(train_images, "train", class_name)
            copy_images(val_images, "val", class_name)
            copy_images(test_images, "test", class_name)

            stats[class_name] = {
                "total": len(images),
                "train": len(train_images),
                "val": len(val_images),
                "test": len(test_images),
            }

            print(
                f"Split {class_name}: "
                f"{len(train_images)} train, "
                f"{len(val_images)} val, "
                f"{len(test_images)} test"
            )

        except Exception as error:
            raise RuntimeError(f"Failed to split class '{class_name}': {error}") from error

    write_report(stats)

    print("\nDataset split complete.")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

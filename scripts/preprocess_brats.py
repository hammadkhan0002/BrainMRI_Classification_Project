"""
Preprocess BraTS 2020 FLAIR MRI slices for 2D tumor classification.

This script uses FLAIR images as inputs and segmentation masks as labels.
It saves meaningful brain-tissue slices as 224x224 PNG images.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


TEST_MODE = True

DATASET_PATH = Path("/Users/hammadkhan/Downloads/MICCAI_BraTS2020_TrainingData")
OUTPUT_DIR = Path("data/processed")
TUMOR_DIR = OUTPUT_DIR / "tumor"
NO_TUMOR_DIR = OUTPUT_DIR / "no_tumor"
REPORT_PATH = OUTPUT_DIR / "preprocessing_report.txt"

OUTPUT_IMAGE_SIZE = (224, 224)

# A slice is kept only if enough pixels contain meaningful MRI signal.
# This avoids saving mostly black background slices.
MIN_BRAIN_PIXEL_RATIO = 0.01


def create_output_dirs():
    """Create output folders if they do not already exist."""
    TUMOR_DIR.mkdir(parents=True, exist_ok=True)
    NO_TUMOR_DIR.mkdir(parents=True, exist_ok=True)


def get_patient_folders(dataset_path):
    """Return sorted patient folders from the BraTS dataset directory."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    patient_folders = sorted(
        folder for folder in dataset_path.iterdir() if folder.is_dir()
    )

    if TEST_MODE:
        return patient_folders[:5]

    return patient_folders


def get_required_files(patient_folder):
    """Return expected FLAIR and segmentation file paths for one patient."""
    patient_name = patient_folder.name
    flair_path = patient_folder / f"{patient_name}_flair.nii"
    seg_path = patient_folder / f"{patient_name}_seg.nii"

    if not flair_path.exists():
        raise FileNotFoundError(f"Missing FLAIR file: {flair_path}")

    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segmentation file: {seg_path}")

    return flair_path, seg_path


def load_nifti(path):
    """Load a NIfTI file and return image data as a NumPy array."""
    try:
        return nib.load(str(path)).get_fdata()
    except Exception as error:
        raise RuntimeError(f"Failed to read NIfTI file {path}: {error}") from error


def is_meaningful_brain_slice(slice_data):
    """Return True if a slice contains enough non-background brain tissue."""
    if slice_data.size == 0:
        return False

    positive_pixels = slice_data > 0
    brain_pixel_ratio = np.count_nonzero(positive_pixels) / slice_data.size

    return brain_pixel_ratio >= MIN_BRAIN_PIXEL_RATIO


def normalize_slice_to_uint8(slice_data):
    """Normalize one MRI slice to 8-bit grayscale."""
    nonzero_values = slice_data[slice_data > 0]

    if nonzero_values.size == 0:
        return np.zeros(slice_data.shape, dtype=np.uint8)

    lower_bound = np.percentile(nonzero_values, 1)
    upper_bound = np.percentile(nonzero_values, 99)

    if upper_bound <= lower_bound:
        return np.zeros(slice_data.shape, dtype=np.uint8)

    clipped = np.clip(slice_data, lower_bound, upper_bound)
    normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
    normalized = normalized * 255

    return normalized.astype(np.uint8)


def save_slice_png(slice_data, output_path):
    """Save one normalized MRI slice as a resized PNG image."""
    normalized_slice = normalize_slice_to_uint8(slice_data)
    image = Image.fromarray(normalized_slice, mode="L")
    image = image.resize(OUTPUT_IMAGE_SIZE, Image.BILINEAR)
    image.save(output_path)


def process_patient(patient_folder):
    """Process all meaningful FLAIR slices for one patient."""
    flair_path, seg_path = get_required_files(patient_folder)

    flair_data = load_nifti(flair_path)
    seg_data = load_nifti(seg_path)

    if flair_data.shape != seg_data.shape:
        raise ValueError(
            f"Shape mismatch for {patient_folder.name}: "
            f"FLAIR {flair_data.shape}, SEG {seg_data.shape}"
        )

    stats = {
        "slices_examined": 0,
        "tumor_saved": 0,
        "no_tumor_saved": 0,
        "discarded": 0,
    }

    for slice_index in range(flair_data.shape[2]):
        stats["slices_examined"] += 1

        flair_slice = flair_data[:, :, slice_index]
        seg_slice = seg_data[:, :, slice_index]

        if not is_meaningful_brain_slice(flair_slice):
            stats["discarded"] += 1
            continue

        has_tumor = np.any(seg_slice > 0)

        if has_tumor:
            output_folder = TUMOR_DIR
            stats["tumor_saved"] += 1
        else:
            output_folder = NO_TUMOR_DIR
            stats["no_tumor_saved"] += 1

        output_name = f"{patient_folder.name}_slice_{slice_index}.png"
        output_path = output_folder / output_name

        save_slice_png(flair_slice, output_path)

    return stats


def write_report(total_patients, totals):
    """Write preprocessing summary report."""
    report_lines = [
        "BraTS 2020 Preprocessing Report",
        "=" * 32,
        f"Total patients processed: {total_patients}",
        f"Total slices examined: {totals['slices_examined']}",
        f"Tumor slices saved: {totals['tumor_saved']}",
        f"No-tumor slices saved: {totals['no_tumor_saved']}",
        f"Discarded slices: {totals['discarded']}",
        f"Output image size: {OUTPUT_IMAGE_SIZE[0]}x{OUTPUT_IMAGE_SIZE[1]}",
        f"TEST_MODE: {TEST_MODE}",
    ]

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main():
    """Run BraTS preprocessing."""
    create_output_dirs()

    patient_folders = get_patient_folders(DATASET_PATH)

    totals = {
        "slices_examined": 0,
        "tumor_saved": 0,
        "no_tumor_saved": 0,
        "discarded": 0,
    }

    processed_patients = 0

    for patient_folder in patient_folders:
        try:
            patient_stats = process_patient(patient_folder)
        except Exception as error:
            print(f"Skipping {patient_folder.name}: {error}")
            continue

        processed_patients += 1

        for key in totals:
            totals[key] += patient_stats[key]

        print(
            f"Processed {patient_folder.name}: "
            f"{patient_stats['tumor_saved']} tumor, "
            f"{patient_stats['no_tumor_saved']} no-tumor, "
            f"{patient_stats['discarded']} discarded"
        )

    write_report(processed_patients, totals)

    print("\nPreprocessing complete.")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

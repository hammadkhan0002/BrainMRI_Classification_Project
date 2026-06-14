"""
Names: Muhammad Hammad Khan and Abdullah Waseem
Project: Brain MRI Classification (BraTS 2020)

Task 1.2 - Dataset Validation
"""

from pathlib import Path

# Dataset location
DATASET_PATH = Path(
    "/Users/hammadkhan/Downloads/MICCAI_BraTS2020_TrainingData"
)

required_suffixes = [
    "_flair.nii",
    "_t1.nii",
    "_t1ce.nii",
    "_t2.nii",
    "_seg.nii",
]

patient_folders = [
    f for f in DATASET_PATH.iterdir() if f.is_dir()
]

print("=" * 50)
print("BraTS 2020 Dataset Validation")
print("=" * 50)

print(f"Total patient folders found: {len(patient_folders)}")

missing_files = []

for patient in patient_folders:

    patient_name = patient.name

    for suffix in required_suffixes:

        expected_file = patient / f"{patient_name}{suffix}"

        if not expected_file.exists():
            missing_files.append(str(expected_file))

if len(missing_files) == 0:
    print("\n✓ All required files are present.")
else:
    print("\nMissing files detected:")
    for file in missing_files:
        print(file)

print("\nValidation complete.")
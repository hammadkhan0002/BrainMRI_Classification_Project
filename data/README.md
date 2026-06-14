# BraTS 2020 Dataset Documentation

## Dataset Information

Dataset Name: BraTS 2020 (Brain Tumor Segmentation Challenge 2020)

Source:
MICCAI BraTS 2020 Dataset

Number of Cases:
371 MRI studies

## Dataset Location

The dataset is stored locally and is not included in this Git repository due to its large size.

Local Path:

/Users/hammadkhan/Downloads/MICCAI_BraTS2020_TrainingData

## Files Available Per Patient

Each patient folder contains:

* FLAIR MRI
* T1 MRI
* T1CE MRI
* T2 MRI
* Segmentation Mask (SEG)

Example:

BraTS20_Training_001_flair.nii
BraTS20_Training_001_t1.nii
BraTS20_Training_001_t1ce.nii
BraTS20_Training_001_t2.nii
BraTS20_Training_001_seg.nii

## Project Usage

This project uses a 2D slice-based classification approach.

Classification Task:

* Tumor Present
* Tumor Absent

The segmentation masks will be used to determine whether a slice contains tumor tissue.

## Notes

The raw dataset is excluded from GitHub using .gitignore because of storage limitations and dataset size.

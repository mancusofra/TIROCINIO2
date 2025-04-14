# Yeast Vacuole Image Classification

This project focuses on the classification of yeast vacuole images into four distinct categories using machine learning and deep learning techniques. The primary goal is to compare multiple ML systems for image classification, evaluating the trade-off between computational cost and classification accuracy.

## Project Overview

Yeast vacuoles can exhibit different morphological patterns depending on various physiological and environmental conditions. In this project, each vacuole image is categorized into one of the following four classes:

- **Multiple**: Vacuoles appear as multiple separate structures within the cell.
- **Condensed**: Vacuoles are tightly packed or shrunk, possibly due to stress or specific treatments.
- **Positive**: Vacuoles show a specific biomarker or staining pattern indicating a positive result.
- **Negative**: No biomarker or staining is detected; vacuoles are considered negative.

## Dataset

The dataset consists of labeled microscopy images of yeast cells with vacuole structures. Each image is cropped to include only a single yeast cell and has a resolution of 80x80 pixels. Each image is annotated with one of the four target classes.

## Feature Engineering

This component extracts a range of morphological features from the vacuole images using both shape-based and texture-based methods. 

[Insert detailed explanation of the extracted features here]

## Machine Learning Approaches

Currently, the project includes an unsupervised prediction model based on **K-Means clustering**. Future updates will integrate and compare multiple supervised and unsupervised learning techniques.

## Acknowledgements



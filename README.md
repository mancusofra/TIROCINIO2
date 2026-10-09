# Yeast Vacuole Image Classification

[![CI](https://github.com/mancusofra/TIROCINIO2/actions/workflows/ci.yml/badge.svg)](https://github.com/mancusofra/TIROCINIO2/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project classifies yeast vacuole microscopy images into four morphological
classes, comparing a hand-crafted-feature pipeline (segmentation → feature
extraction → clustering-assisted filtering → classification) against the
computational cost of each stage.

Each image is cropped to a single yeast cell (80x80 px) and labeled with one of:

- **Multiple** — vacuoles appear as multiple separate structures within the cell
- **Condensed** — vacuoles are tightly packed/shrunk
- **Positive** — a biomarker/staining pattern is present
- **Negative** — no biomarker/staining is detected

## Repository layout

```
yeast_vacuole_pipeline/ <- the pipeline (see below)
requirements.txt       <- Python dependencies
```

## Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # runtime (includes PyTorch)
pip install -r requirements-dev.txt    # pytest + ruff
```

## Development

```
ruff check . && ruff format --check .
pytest
```

The test suite covers data processing, fuzzy filtering, geometric features and
the segmentation code (metrics, dataset, a training step on synthetic images),
and runs on every push via GitHub Actions.

## yeast_vacuole_pipeline

The current, complete pipeline. Run interactively via its menu:

```
python -m yeast_vacuole_pipeline.main
```

Modules:

| Module | Purpose |
|---|---|
| `unet_segmentation/` | Cell/vacuole segmentation with a U-Net (`segmentation_models_pytorch`, EfficientNet-B7 encoder): `dataset.py` (image/mask pairing), `metrics.py` (Dice, IoU, BCE+Dice loss), `training.py` (training with early stopping, evaluation, plots), `load_model.py` (inference) |
| `features_extraction/` | Builds the masked/grayscale dataset from segmented images and extracts per-cell features via `extractors/` (geometric, gray-level histogram, Haralick texture, Hu moments, LBP, Zernike moments) |
| `data_processing/` | `load_data` (CSV features → DataFrame), `shuffle_data` (injects a controlled % of mislabeled rows for testing filtering), `df_compare` (diffing/mismatch helpers) |
| `fuzzy_km/` | Fuzzy c-means clustering (`fuzzy_clustering`) used, together with `unsupervised_assisted_filtering`, to flag and drop low-confidence/likely-mislabeled samples before training |
| `random_forest/` | `fit_random_forest` (training) and `model_accuracy` (evaluation) — the final supervised classifier, trained on raw vs. filtered data to measure the effect of unsupervised-assisted filtering |
| `visualizer/` | `pca_plot.py` — dimensionality-reduction scatter plots of the extracted feature space |

### Data

Raw images, masks and extracted features are not versioned (see `.gitignore`:
`Data/`, `Features/`). The code expects a `Data/` folder (with `Original_images/`,
`Mask/`, `DataSet/`, `Features/`, `Model/`) alongside the scripts that reference it.

Paths are resolved relative to the package (see `PIPELINE_DIR` in `main.py`), so the
pipeline works from any checkout location as long as `yeast_vacuole_pipeline/Data/` exists.

### Known limitations / possible future work

- Feature extraction currently gives a shape/texture-based description that is
  only moderately discriminative: the fuzzy k-means confidence for a *correct*
  label isn't much higher than for an *incorrect* one.
- Feature extraction should error out clearly if run before the images have been
  segmented/pre-processed, instead of failing implicitly downstream.
- Possible directions: add more features, refine the existing ones, or extract
  deep features from a pretrained CNN instead of/alongside hand-crafted ones.

## Acknowledgements

The segmentation setup (U-Net with a pretrained EfficientNet-B7 encoder from
`segmentation_models_pytorch`, trained with a BCE + Dice loss) was inspired by
public Kaggle notebooks on the LGG Brain MRI Segmentation dataset, then
reimplemented for 80x80 yeast cell images.

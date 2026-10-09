# Yeast Vacuole Image Classification

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
YeastVacuolePipeline/   <- the pipeline (see below)
requirements.txt        <- Python dependencies
```

## YeastVacuolePipeline

The current, complete pipeline. Run interactively via its menu:

```
python -m YeastVacuolePipeline.Main
```

Modules:

| Module | Purpose |
|---|---|
| `UNETSegmentation/` | Cell/vacuole segmentation. Two model variants: `UNETSegmentation.py` (Keras/TensorFlow U-Net) and `UNETorch.py` + `LoadModel.py` (PyTorch U-Net via `segmentation_models_pytorch`, EfficientNet-B7 encoder) |
| `FeaturesExtraction/` | Builds the masked/grayscale dataset from segmented images and extracts per-cell features via `Extractors/` (geometric, gray-level histogram, Haralick texture, Hu moments, LBP, Zernike moments) |
| `DataProcessing/` | `LoadData` (CSV features → DataFrame), `ShuffleData` (injects a controlled % of mislabeled rows for testing filtering), `DFCompare` (diffing/mismatch helpers) |
| `FuzzyKM/` | Fuzzy c-means clustering (`FuzzyClustering`) used, together with `UnsupervisedAssistedFiltering`, to flag and drop low-confidence/likely-mislabeled samples before training |
| `RandomForestClassifier/` | `FitRandomForest` (training) and `ModelAccuracy` (evaluation) — the final supervised classifier, trained on raw vs. filtered data to measure the effect of unsupervised-assisted filtering |
| `Visualizer/` | `PCA.py` — dimensionality-reduction scatter plots of the extracted feature space |

### Data

Raw images, masks and extracted features are not versioned (see `.gitignore`:
`Data/`, `Features/`). The code expects a `Data/` folder (with `Original_images/`,
`Mask/`, `DataSet/`, `Features/`, `Model/`) alongside the scripts that reference it.

Paths are resolved relative to the package (see `PIPELINE_DIR` in `Main.py`), so the
pipeline works from any checkout location as long as `YeastVacuolePipeline/Data/` exists.

### Known limitations / possible future work

- Feature extraction currently gives a shape/texture-based description that is
  only moderately discriminative: the fuzzy k-means confidence for a *correct*
  label isn't much higher than for an *incorrect* one.
- Feature extraction should error out clearly if run before the images have been
  segmented/pre-processed, instead of failing implicitly downstream.
- Possible directions: add more features, refine the existing ones, or extract
  deep features from a pretrained CNN instead of/alongside hand-crafted ones.

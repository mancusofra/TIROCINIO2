import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from yeast_vacuole_pipeline.unet_segmentation.load_model import load_model, predict

# Features Extraction Modules
from .extractors.geometric_features import extract_geometric_features
from .extractors.gray_hist_features import extract_gray_hist_features
from .extractors.haralick_features import extract_haralick_features
from .extractors.hu_moments_features import extract_hu_moments
from .extractors.lbp_features import extract_lbp_features
from .extractors.zernike_features import extract_zernike_moments


def file_extraction(input_dir, Verbose=False):
    """
    Lists the masked and grayscale .tif files of a dataset built by
    mask_dataset_maker/gray_dataset_maker (input_dir/MaskedImages and
    input_dir/GrayImages), recursively, one subfolder per class.

    Args:
        input_dir (str): Dataset root, containing "MaskedImages" and "GrayImages".
        Verbose (bool): If True, prints how many images were found.

    Returns:
        tuple: (mask_files, image_files), matching lists of file paths.

    Raises:
        ValueError: If the number of mask files and image files differs.
    """
    image_dir = os.path.join(input_dir, "GrayImages")
    mask_dir = os.path.join(input_dir, "MaskedImages")
    image_files = []
    mask_files = []

    for root, dirs, files in os.walk(mask_dir):
        for file in files:
            if file.endswith(".tif"):
                mask_files.append(os.path.join(root, file))

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".tif"):
                image_files.append(os.path.join(root, file))

    if len(mask_files) != len(image_files):
        raise ValueError("Masks and grayscale images do not match.")

    elif Verbose:
        print(f"Images found: {len(mask_files)}")

    return mask_files, image_files


def features_extraction(gray_images, masked_images, features_dir="./Data/Features", params=None):
    """
    Extracts the full feature set for each image pair and writes one CSV per
    sample under features_dir/<class>/<sample_name>.csv.

    For each (grayscale, mask) pair, runs all six extractors — geometric
    (on the mask), Hu moments and Zernike moments (on the mask), Haralick,
    LBP and gray-histogram (on the grayscale image) — and concatenates every
    value into a single column, one value per line.

    Args:
        gray_images (list): Paths to grayscale (masked-RGB) .tif images.
        masked_images (list): Paths to the matching binary mask .tif images.
        features_dir (str): Output directory; one CSV subfolder per class.
        params (dict, optional): Per-extractor keyword overrides, e.g.
            {"zernike": {...}, "lpb": {...}} for extract_zernike_moments /
            extract_lbp_features.

    Returns:
        str: features_dir, unchanged (for chaining).
    """
    for gray_path, masked_path in tqdm(
        zip(gray_images, masked_images),
        total=len(gray_images),
        desc=f"Extracting {features_dir.split('/')[-2]}",
    ):
        features = ""
        # Convert files to 8-bit format required for feature extraction operations
        gray_image = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        masked_image = cv2.imread(masked_path, cv2.IMREAD_GRAYSCALE)

        # Define output path for CSV files
        dir_name = f"{features_dir}/{masked_path.split('/')[-2]}/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name + masked_path.split("/")[-1][0:-4] + ".csv"

        # Geometric/shape features run on gray_image: background is already
        # zeroed out by gray_dataset_maker, so contours match the cell outline.
        geo_features = extract_geometric_features(gray_image)

        for val in geo_features.values():
            features += str(val) + "\n"

        # Hu invariant moments describe the shape of the object independently of rotation, scale, and translation.
        for val in extract_hu_moments(masked_image):
            features += str(val) + "\n"

        # Zernike moments capture symmetry and shape complexity.
        if params and ("zernike" in params):
            zernike_params = params["zernike"]
            if isinstance(zernike_params, dict):
                zernike_features = extract_zernike_moments(masked_image, **zernike_params)
            else:
                raise ValueError("Invalid parameters for Zernike features.")
        else:
            zernike_features = extract_zernike_moments(masked_image)

        for val in zernike_features:
            features += str(val) + "\n"

        # Texture features (Haralick, LBP) run on gray_image too: they depend on
        # local pixel-intensity variation, which the binary mask alone can't capture.
        for val in extract_haralick_features(gray_image):
            features += str(val) + "\n"

        if params and ("lpb" in params):
            lpb_params = params["lpb"]
            if isinstance(lpb_params, dict):
                lpb_features = extract_lbp_features(gray_image, **lpb_params)
            else:
                raise ValueError("Invalid parameters for LBP features.")

        else:
            lbp_features = extract_lbp_features(gray_image)

        for val in extract_lbp_features(gray_image):
            features += str(val) + "\n"

        # Extract features based on grayscale histogram.
        for val in extract_gray_hist_features(gray_image):
            features += str(val) + "\n"

        with open(file_name, "w", newline="") as file:
            file.write(features)

    return features_dir


def mask_dataset_maker(input_dir, output_dir, masked_dir, model_path, verbose=False, test=False):
    """
    Builds output_dir/MaskedImages by reusing existing manual masks where
    available and predicting the rest with the U-Net segmentation model.

    For each .tif under input_dir (one subfolder per class), copies the
    matching mask from masked_dir if present, otherwise runs the model on
    the raw image to predict one.

    Args:
        input_dir (str): Root of the raw images, one subfolder per class.
        output_dir (str): Dataset root to write "MaskedImages" into
            (suffixed with "_test" when test=True).
        masked_dir (str): Root of existing manually-annotated masks, if any.
        model_path (str): Path to the U-Net model weights (see unet_segmentation.load_model).
        verbose (bool): If True, prints the number of files written per class.
        test (bool): If True, appends "_test" to output_dir.
    """
    model = load_model(model_path)

    if test:
        output_dir = output_dir + "_test"

    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for subdir in tqdm(subdirs):
        output_mask_path = f"{output_dir}/MaskedImages/{subdir}/"
        if not os.path.exists(output_mask_path):
            os.makedirs(output_mask_path)

        subdir_path = os.path.join(input_dir, subdir)
        masked_subdir = os.path.join(masked_dir, subdir)

        tif_files = [f for f in os.listdir(subdir_path) if f.endswith(".tif")]
        masked_tif_files = [f for f in os.listdir(masked_subdir) if f.endswith(".tif")]

        for f in tif_files:
            if f not in masked_tif_files:
                full_path = os.path.join(subdir_path, f)
                predicted_f_mask = predict(model, full_path)
                # show_image(predicted_f_mask)

                cv2.imwrite(f"{output_mask_path}/{f}", predicted_f_mask * 255)

            else:
                manual_f_mask = cv2.imread(f"{masked_subdir}/{f}", cv2.IMREAD_COLOR)
                cv2.imwrite(f"{output_mask_path}/{f}", manual_f_mask)

        # print(f"{count1} {count2}")
        num_files = sum([len(files) for _, _, files in os.walk(output_mask_path)])
        if verbose:
            print(f"Number of files in {output_mask_path}: {num_files}")


def gray_dataset_maker(input_dir, rgb_dir, verbose=False, test=False):
    """
    Builds the "GrayImages" set by masking each RGB image with its binary
    mask, so only the segmented cell remains (background zeroed out).

    Reads masks from input_dir/MaskedImages and matching RGB images from
    rgb_dir (paired by sorted filename order within each class subfolder),
    and writes the masked-RGB result to input_dir/GrayImages.

    Args:
        input_dir (str): Dataset root containing "MaskedImages" (and where
            "GrayImages" will be written). Note: despite the name, this is
            the same output_dir used by mask_dataset_maker, not a grayscale
            input.
        rgb_dir (str): Root of the original RGB images, one subfolder per class.
        verbose (bool): Unused here, kept for interface consistency.
        test (bool): If True, reads/writes the "_test" variant of input_dir.
    """
    if test:
        output_dir = input_dir + "_test/GrayImages"
        input_dir = input_dir + "_test/MaskedImages"
    else:
        output_dir = input_dir + "/GrayImages"
        input_dir = input_dir + "/MaskedImages"

    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for subdir in subdirs:
        output_gray_dir = f"{output_dir}/{subdir}"
        input_gray_dir = f"{input_dir}/{subdir}"
        rgb_subdir = f"{rgb_dir}/{subdir}"

        if not os.path.exists(output_gray_dir):
            os.makedirs(output_gray_dir)

        tif_files = [f for f in os.listdir(input_gray_dir) if f.endswith(".tif")]
        rgb_tif_files = [f for f in os.listdir(rgb_subdir) if f.endswith(".tif")]

        # Assumes filenames correspond after sorting
        for mask_name, rgb_name in zip(sorted(tif_files), sorted(rgb_tif_files)):
            mask_path = os.path.join(input_gray_dir, mask_name)
            rgb_path = os.path.join(rgb_subdir, rgb_name)

            # Load images
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            rgb = cv2.imread(rgb_path)

            # Make sure dimensions match
            if mask.shape != rgb.shape[:2]:
                print(f"Dimension mismatch: {mask_name} vs {rgb_name}")
                continue

            # Build a binary mask (values 0 and 1)
            binary_mask = (mask > 0).astype(np.uint8)

            # Expand to 3 channels to mask the RGB image
            binary_mask_3c = cv2.merge([binary_mask] * 3)

            # Apply the mask to the RGB image
            masked_rgb = cv2.bitwise_and(rgb, rgb, mask=binary_mask)

            # Save
            cv2.imwrite(f"{output_gray_dir}/{mask_name}", masked_rgb)


def full_dataset_maker(
    input_dir, output_dir, masked_dir, features_dir, model_path, verbose=False, test=False
):
    """
    End-to-end dataset build: segmentation masks -> masked-RGB images ->
    per-sample feature CSVs.

    Chains mask_dataset_maker, gray_dataset_maker, file_extraction and
    features_extraction in sequence, applying the same test/train suffixing
    throughout.

    Args:
        input_dir (str): Root of the raw RGB images, one subfolder per class.
        output_dir (str): Dataset root to build ("MaskedImages", "GrayImages").
        masked_dir (str): Root of existing manually-annotated masks, if any.
        features_dir (str): Output directory for the extracted feature CSVs.
        model_path (str): Path to the U-Net model weights used to predict
            missing masks.
        verbose (bool): If True, prints progress details from the sub-steps.
        test (bool): If True, builds the "_test" variant end to end.

    Returns:
        str: The features directory actually used (features_dir, or
        features_dir + "_test").
    """
    mask_dataset_maker(input_dir, output_dir, masked_dir, model_path, verbose=verbose, test=test)
    gray_dataset_maker(output_dir, input_dir, verbose=verbose, test=test)

    if test:
        features_dir = features_dir + "_test"
        output_dir = output_dir + "_test"

    mask_files, image_files = file_extraction(output_dir, Verbose=verbose)
    features_dir = features_extraction(image_files, mask_files, features_dir=features_dir)
    return features_dir


if __name__ == "__main__":
    # Data/ lives next to the yeast_vacuole_pipeline package, regardless of
    # machine or user (this file is at yeast_vacuole_pipeline/features_extraction/).
    DATA_DIR = (Path(__file__).resolve().parent.parent / "Data").as_posix()

    masked_dir = f"{DATA_DIR}/Mask"
    input_dir = f"{DATA_DIR}/Original_images/Train_annotated"
    input_dir_test = f"{DATA_DIR}/Original_images/test"
    output_dir = f"{DATA_DIR}/DataSet"
    model_path = f"{DATA_DIR}/Model/Weights.pt"
    features_dir = f"{DATA_DIR}/Features"

    full_dataset_maker(
        input_dir, output_dir, masked_dir, features_dir, model_path, verbose=False, test=False
    )
    full_dataset_maker(
        input_dir_test, output_dir, masked_dir, features_dir, model_path, verbose=False, test=True
    )

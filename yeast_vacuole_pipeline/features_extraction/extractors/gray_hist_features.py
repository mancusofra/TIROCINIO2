import numpy as np
from scipy.stats import kurtosis, skew


def extract_gray_hist_features(
    img_gray, features_vector=["mean", "std", "kurtosis", "skewness", "entropy"]
):
    """
    Extracts statistical features from the grayscale intensity histogram.

    These features describe the global distribution of pixel intensities in
    the image, capturing brightness, contrast, and overall texture
    characteristics. Pixel values are rescaled to [0, 255] first if the image
    is normalized (0-1), then flattened for the global statistics:
        - mean: average brightness of the image.
        - std: standard deviation, i.e. contrast/spread of intensities.
        - kurtosis: how heavy/light the tails of the distribution are (sharpness/flatness).
        - skewness: asymmetry of the histogram (left/right bias).
        - entropy: measure of randomness or texture complexity in the image.

    Args:
        img_gray: Grayscale image, normalized ([0, 1]) or 8-bit ([0, 255]).
        features_vector (list): Subset of the above feature names to keep in the output.

    Returns:
        list: [mean, std, kurtosis, skewness, entropy] for the image.
    """
    if img_gray.max() <= 1.0:
        img_gray = (img_gray * 255).astype(np.uint8)

    pixels = img_gray.flatten()

    mean_val = np.mean(pixels)
    std_val = np.std(pixels)
    kurt = kurtosis(pixels, fisher=True)
    asym = skew(pixels)

    hist, _ = np.histogram(pixels, bins=256, range=(0, 256), density=True)
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

    features = []

    # Create a mapping of feature names to their computed values
    # This allows for easy retrieval of feature values based on the requested features.
    feature_map = {
        "mean": mean_val,
        "std": std_val,
        "kurtosis": kurt,
        "skewness": asym,
        "entropy": entropy,
    }

    for feature in features_vector:
        if feature in feature_map:
            features.append(feature_map[feature])

    return [mean_val, std_val, kurt, asym, entropy]

import numpy as np
from scipy.stats import kurtosis, skew

def extract_gray_hist_features(img_gray):
    # Extract statistical features from the grayscale intensity histogram.
    # These features describe the global distribution of pixel intensities in the image
    # and are useful for capturing brightness, contrast, and overall texture characteristics.
    #
    # 1. Ensure pixel values are in the [0, 255] range. If the image is normalized (0–1),
    #    rescale it to 8-bit format.
    # 2. Flatten the image for easier computation of global statistics.
    #
    # Extracted features:
    # - Mean: average brightness of the image.
    # - Standard deviation: contrast or spread of intensities.
    # - Kurtosis: how heavy/light the tails of the distribution are (sharpness/flatness).
    # - Skewness: asymmetry of the histogram (left/right bias).
    # - Entropy: measure of randomness or texture complexity in the image.
    #
    # These 5 values form a compact descriptor for grayscale intensity distribution.

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

    return [mean_val, std_val, kurt, asym, entropy]
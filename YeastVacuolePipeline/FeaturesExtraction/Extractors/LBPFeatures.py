import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(gray,
    P = 8,
    R = 1):
    """
    Computes Local Binary Pattern (LBP) features, a simple yet powerful
    descriptor for local texture information in an image.

    Thresholds the neighborhood of each pixel and encodes the result as a
    binary number, using P sampling points on a circle of radius R, with the
    "uniform" method, which focuses on patterns with minimal transitions
    (e.g. edges, spots). The resulting LBP image is converted into a
    normalized histogram of pattern occurrences (density=True), giving a
    texture feature vector that is invariant to image size.

    Args:
        gray: Grayscale image.
        P (int): Number of circularly symmetric sampling points.
        R (int): Radius of the sampling circle.

    Returns:
        np.ndarray: Normalized histogram of LBP pattern occurrences.
    """
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
    return hist
import numpy as np
from skimage.feature import local_binary_pattern


def extract_lbp_features(gray, 
    P = 8,
    R = 1):   
    # Compute Local Binary Pattern (LBP) features, which are simple yet powerful descriptors
    # for capturing local texture information in an image.
    #
    # The LBP operator works by thresholding the neighborhood of each pixel and encoding the result
    # as a binary number. In this case, we use 16 sampling points (P=16) on a circle of radius 2 (R=2),
    # with the "uniform" method, which focuses on patterns with minimal transitions (e.g., edges, spots).
    #
    # The resulting LBP image is then converted into a histogram of pattern occurrences.
    # We normalize the histogram (density=True) to get a texture feature vector that is invariant to image size.
    #
    # This vector summarizes the frequency of texture patterns across the image.
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
    return hist
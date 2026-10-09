import numpy as np
from skimage.measure import moments, moments_hu


def extract_hu_moments(gray):
    """
    Computes the seven Hu moments, which capture the shape of objects and are
    invariant to translation, rotation, and scaling, making them ideal for
    tasks like object recognition or shape matching.

    Hu moments can span a very large range of values (e.g. from 1e-9 to
    1e+3), so a logarithmic transform is applied to compress their scale,
    making the features easier to compare and avoiding issues in downstream
    tasks like classification: -sign * log10(abs(moment)), with a small
    epsilon (1e-10) added to prevent taking log(0).

    Args:
        gray: Grayscale (or binary) image.

    Returns:
        np.ndarray: The 7 log-transformed Hu moments.
    """
    mom = moments(gray)
    hu = moments_hu(mom)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu_log

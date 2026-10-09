from mahotas.features import haralick

def extract_haralick_features(gray):
    """
    Computes Haralick texture features based on the gray-level co-occurrence
    matrix (GLCM).

    These features capture texture properties like contrast, correlation,
    entropy, and homogeneity, useful for characterizing patterns, surfaces,
    or regions in an image. `haralick` computes them in multiple directions
    (e.g. 0°, 45°, 90°, 135°); the mean across directions is taken to get a
    single representative feature vector.

    Args:
        gray: Grayscale image.

    Returns:
        np.ndarray: 13-dimensional feature vector summarizing image texture.
    """
    return haralick(gray).mean(axis=0)
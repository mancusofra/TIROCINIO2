import numpy as np
from mahotas.features import zernike_moments

def extract_zernike_moments(gray,
    radius = 30,
    degree = 8):
    """
    Computes Zernike moments, advanced shape descriptors based on orthogonal
    polynomials. They capture both the geometry and symmetry of a shape and
    are invariant to rotation, making them highly effective for pattern
    recognition and image analysis.

    Args:
        gray: Binary or grayscale image.
        radius (int): Maximum distance from the center to consider.
        degree (int): Level of detail (higher degrees capture finer structures).

    Returns:
        np.ndarray: Zernike moment feature vector.

    Raises:
        ValueError: If the resulting feature vector is all zeros, which
            usually means the image doesn't contain a recognizable shape
            within the given radius.
    """
    zernike_features = zernike_moments(gray, radius=radius, degree=degree)
    if np.all(zernike_features == 0):
        raise ValueError("Zernike moment is null for an image.")
        
    return zernike_features
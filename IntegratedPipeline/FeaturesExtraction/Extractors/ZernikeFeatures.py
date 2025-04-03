import numpy as np
from mahotas.features import zernike_moments

def extract_zernike_moments(gray):
    # Compute Zernike Moments, which are advanced shape descriptors based on orthogonal polynomials.
    # They capture both the geometry and symmetry of a shape and are invariant to rotation,
    # making them highly effective for pattern recognition and image analysis.
    #
    # 'radius' --> defines the maximum distance from the center to consider,
    # 'degree' --> controls the level of detail (higher degrees capture finer structures).
    #
    # If the resulting feature vector is all zeros, it likely means the image doesn't contain
    # a recognizable shape within the given radius, so we raise an error to flag this edge case.
    
    zernike_features = zernike_moments(gray, radius=30, degree=8)
    if np.all(zernike_features == 0):
        raise ValueError("Zernike moment is null for an image.")
        
    return zernike_features
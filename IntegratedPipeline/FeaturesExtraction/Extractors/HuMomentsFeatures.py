import numpy as np
from skimage.measure import moments, moments_hu

def extract_hu_moments(gray):
    # Compute Hu Moments, which are seven values derived from image moments.
    # They capture the shape of objects and are invariant to translation, rotation, and scaling,
    # making them ideal for tasks like object recognition or shape matching.
    #
    # Since Hu Moments can span a very large range of values (e.g., from 1e-9 to 1e+3),
    # we apply a logarithmic transformation to compress their scale. This makes the features
    # easier to compare and helps avoid issues in downstream tasks like classification.
    #
    # The transformation is done using: -sign * log10(abs(moment)), and a small epsilon (1e-10)
    # is added to prevent taking log(0), which is undefined.
    
    mom = moments(gray)
    hu = moments_hu(mom)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu_log
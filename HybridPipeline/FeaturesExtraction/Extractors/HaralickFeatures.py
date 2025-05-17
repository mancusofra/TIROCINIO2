from mahotas.features import haralick

def extract_haralick_features(gray):
    # Compute Haralick texture features based on the gray-level co-occurrence matrix (GLCM).
    # These features capture texture properties like contrast, correlation, entropy, and homogeneity,
    # which are useful for characterizing patterns, surfaces, or regions in an image.
    #
    # The `haralick` function computes these features in multiple directions (e.g., 0°, 45°, 90°, 135°).
    # To get a single representative feature vector, we take the mean across all directions.
    #
    # This results in a 13-dimensional feature vector that summarizes the image's texture characteristics.
    
    return haralick(gray).mean(axis=0)
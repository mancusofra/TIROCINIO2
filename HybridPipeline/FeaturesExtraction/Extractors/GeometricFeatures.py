import cv2
import numpy as np


class ContourNotFoundError(Exception):
    pass


def extract_geometric_features(gray,
    features_vector = ["total_area", "total_perimeter", "eccentricity", "circularity", "solidity", "extent", "mean_radius"]):
    # Extract geometric features from a binary or grayscale image.
    
    # The function analyzes the largest external contour found in the image
    # and computes the following shape descriptors:

    # - total_area: Area of the largest contour.
    # - total_perimeter: Perimeter (arc length) of the largest contour.
    # - eccentricity: How elongated the shape is (requires at least 5 contour points).
    # - circularity: Measure of how circular the shape is (1.0 = perfect circle).
    # - solidity: Ratio between contour area and its convex hull area.
    # - extent: Ratio between contour area and its bounding rectangle area.
    # - mean_radius: Average distance from contour points to the centroid.

    # Raises:
    # - ContourNotFoundError: If no contours are found or the contour is too small for ellipse fitting.

    features = {}
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True) # True for closed contour vedere se ha senso tenerla false ma non credo
        features["total_area"] = area
        features["total_perimeter"] = perimeter

        if len(largest_contour) >= 5:  # fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(largest_contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes) / 2.0  # a
            minor_axis = min(axes) / 2.0  # b
            eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2))
            features["eccentricity"] = eccentricity
        else:
            raise ContourNotFoundError("The contour has fewer than 5 points (unable to compute eccentricity).")

        # Circularity: 1 for a perfect circle, <1 for irregular shapes
        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 0
        features["circularity"] = circularity

        # Solidity: ratio of contour area to convex hull area
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0
        features["solidity"] = solidity

        # Extent: ratio of contour area to bounding rectangle area
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect_area = w * h
        extent = area / rect_area if rect_area != 0 else 0
        features["extent"] = extent

        # Mean Radius: average distance from contour points to the centroid
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            distances = [np.sqrt((point[0][0] - centroid_x)**2 + (point[0][1] - centroid_y)**2) for point in largest_contour]
            mean_radius = np.mean(distances)
            features["mean_radius"] = mean_radius
    else:
        raise ContourNotFoundError("The image does not contain any detectable contours.")

    for feature in list(features.keys()):
        if feature not in features_vector:
            del features[feature]

    return features
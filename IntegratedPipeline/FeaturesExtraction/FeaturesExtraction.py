import os, cv2

# Features Extraction Modules
from .Extractors.GeometricFeatures import extract_geometric_features
from .Extractors.HuMomentsFeatures import extract_hu_moments
from .Extractors.ZernikeFeatures import extract_zernike_moments
from .Extractors.HaralickFeatures import extract_haralick_features
from .Extractors.LBPFeatures import extract_lbp_features
from .Extractors.GrayHistFeatures import extract_gray_hist_features


def file_extraction(input_dir, Verbose=False):
    image_dir = os.path.join(input_dir, "GrayImages")
    mask_dir = os.path.join(input_dir, "MaskedImages")
    image_files = []
    mask_files = []

    for root, dirs, files in os.walk(mask_dir):
        for file in files:
            if file.endswith(".tif"):
                mask_files.append(os.path.join(root, file))

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".tif"):
                image_files.append(os.path.join(root, file))

    if len(mask_files) != len(image_files):
        raise ValueError("Masks and grayscale images do not match.")

    elif Verbose:
        print(f"Images found: {len(mask_files)}")

    return mask_files, image_files


def features_extraction(gray_images, masked_images, features_dir="./Data/Features/", params = None):
    for gray_path, masked_path in zip(gray_images, masked_images):

        features = ""
        # Convert files to 8-bit format required for feature extraction operations
        gray_image = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        masked_image = cv2.imread(masked_path, cv2.IMREAD_GRAYSCALE)

        # Define output path for CSV files
        dir_name = f"{features_dir}{masked_path.split('/')[-2]}/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name + masked_path.split('/')[-1][0:-4] + ".csv"

        # This code extracts geometric features from the images to analyze the shape of the objects.
        # Since these features are based on object structure and shape, we use binary images (masks),
        # which provide a clear representation without color interference.

        # Extract geometric features based on object shape and structure.        
        geo_features = extract_geometric_features(gray_image)
        
        for val in geo_features.values():
            features += str(val) + "\n"

        # Hu invariant moments describe the shape of the object independently of rotation, scale, and translation.
        for val in extract_hu_moments(masked_image):
            features += str(val) + "\n"

        # Zernike moments capture symmetry and shape complexity.
        if params and ("zernike" in params):
            zernike_params = params["zernike"]
            if isinstance(zernike_params, dict):
                zernike_features = extract_zernike_moments(masked_image, **zernike_params)
            else:
                raise ValueError("Invalid parameters for Zernike features.")
        else:
            zernike_features = extract_zernike_moments(masked_image)


        for val in zernike_features:
            features += str(val) + "\n"

        # Extract texture-based features from the image.
        # Since these features depend on local pixel variation and surface patterns of the object,
        # we use grayscale images to better capture intensity differences without color influence.

        # Haralick features analyze pixel co-occurrence to describe image texture.
        for val in extract_haralick_features(gray_image):
            features += str(val) + "\n"

        # LBP (Local Binary Pattern) captures local texture patterns based on intensity differences among neighboring pixels.
        if params and ("lpb" in params):
            lpb_params = params["lpb"]
            if isinstance(lpb_params, dict):
                lpb_features = extract_lbp_features(gray_image, **lpb_params)
            else:
                raise ValueError("Invalid parameters for LBP features.")
        
        else:
            lbp_features = extract_lbp_features(gray_image)

        for val in extract_lbp_features(gray_image):
            features += str(val) + "\n"

        # Extract features based on grayscale histogram.
        for val in extract_gray_hist_features(gray_image):
            features += str(val) + "\n"

        with open(file_name, 'w', newline='') as file:
            file.write(features)

    return features_dir


if __name__ == "__main__":
    dataset_path = "../DataSet"
    mask_files, image_files = file_extraction(dataset_path, Verbose=True)

    features_dir = features_extraction(image_files, mask_files)
    print(features_dir)

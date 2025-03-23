import os

def process_images(directory):
    print(f"Processing images in directory: {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif'):
                file_path = os.path.join(root, file)
                class_name = os.path.basename(root)
                print(f"File: {file}, Class: {class_name}")

# Example usage
process_images("/home/francesco/TIROCINIO2/IntegratedPipeline/KilledBorder")
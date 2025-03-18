import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def visualize_images(base_path, num_images_per_folder=3):
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    fig, axes = plt.subplots(len(folders), num_images_per_folder, figsize=(15, 5 * len(folders)))
    
    for row, folder in enumerate(folders):
        folder_path = os.path.join(base_path, folder)
        images = [img for img in os.listdir(folder_path) if img.endswith('.tif')]
        
        for col in range(num_images_per_folder):
            if col < len(images):
                img_path = os.path.join(folder_path, images[col])
                img = mpimg.imread(img_path)
                axes[row, col].imshow(img)
                axes[row, col].set_title(folder)
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_path = '/home/francesco/TIROCINIO2/MakeMask'
    visualize_images(base_path)
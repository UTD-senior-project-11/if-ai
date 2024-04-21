import cv2
import numpy as np
import os

src_dir = os.getcwd()

# Specify target size for resizing images
width = 100
height = 100

def resize_image(image, target_size):
    """Resize the image to the target size (tuple: width, height)."""
    return cv2.resize(image, target_size)

def load_images_from_folder(folder):
    """Load images from the specified folder and resize them to a target size."""
    images = []
    labels = []  

    # Iterate through "banned" and "allowed" subfolders
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)

        # Assign label based on subfolder name
        if os.path.isdir(subfolder_path):
            label = 1 if subfolder == "allowed" else 0

            # Grayscale the images
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize image
                if image is not None:
                    image = resize_image(image, (width, height))
                    images.append(image)
                    labels.append(label)

    return np.array(images), np.array(labels)


# Specify the path to the folder containing images
folder_path = src_dir + "/../build/dataset/"

# Load images and corresponding labels from the folder
images, labels = load_images_from_folder(folder_path)

# Normalize pixel values to the range [0, 255]
images = images.astype("float32") / 255.0

# Print the shape of the images array and the labels array
#print("Images shape:", images.shape)
#print("Labels shape:", labels.shape)

# Save the images to a numpy file for loading later on
np.save(f'{folder_path}images.npy', images)
np.save(f'{folder_path}labels.npy', labels)

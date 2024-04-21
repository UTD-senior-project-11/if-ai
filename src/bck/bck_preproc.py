import cv2
import numpy as np
import os

# Define the fixed size for resizing
target_size = (100, 100)

# Function to preprocess and load images
def load_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, target_size)  # Resize image
        img = img / 255.0  # Normalize pixel intensities
        images.append(img)
        labels.append(label)
    return images, labels

# Load "banned" images
banned_images, banned_labels = load_images("banned", label=0)

# Load "allowed" images
allowed_images, allowed_labels = load_images("allowed", label=1)

# Convert lists to NumPy arrays
banned_images = np.array(banned_images)
allowed_images = np.array(allowed_images)
banned_labels = np.array(banned_labels)
allowed_labels = np.array(allowed_labels)

# Concatenate "banned" and "allowed" images and labels
all_images = np.concatenate((banned_images, allowed_images), axis=0)
all_labels = np.concatenate((banned_labels, allowed_labels), axis=0)

# Shuffle the data (if needed)
shuffle_indices = np.random.permutation(len(all_images))
all_images = all_images[shuffle_indices]
all_labels = all_labels[shuffle_indices]

# Save the NumPy arrays
np.save("images.npy", all_images)
np.save("labels.npy", all_labels)

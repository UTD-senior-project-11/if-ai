import numpy as np
import cv2
import sys

# FLANN_INDEX_LSH = 6  # Not used in the corrected script

img1 = cv2.imread(sys.argv[1], 0)

# Initialize the BRISK detector
brisk = cv2.BRISK_create()

# Detect keypoints and compute descriptors for the first image
kp1, des1 = brisk.detectAndCompute(img1, None)

# FLANN parameters (used for FLANN based matcher)
# FLANN_INDEX_LSH is not used in this version
index_params = dict(algorithm = cv2.FlannBasedMatcher_INDEX_LSH,  # Use FLANN_INDEX_LSH for older versions
                    table_number = 6,  # These parameters are specific to LSH algorithm
                    key_size = 12,
                    multi_probe_level = 1)
search_params = dict(checks=50)

# Initialize FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Loop through each image in the provided arguments
for filename in sys.argv[2:]:
    img2 = cv2.imread(filename, 0)
    print("Detecting and computing {0}".format(filename))

    # Detect keypoints and compute descriptors for the current image
    kp2, des2 = brisk.detectAndCompute(img2, None)

    # Add the descriptors of the current image to the FLANN matcher
    flann.add(np.float32(des2))

print("Number of descriptors added: ", len(flann.getTrainDescriptors()))  # Verify added descriptors

# Training is not needed for FLANN with the provided descriptors
# flann.train()

print("Matching...")
# Match descriptors of the first image with all added descriptors
matches = flann.knnMatch(np.float32(des1), k=2)

# matches variable will contain the matching results
# You can further process matches based on your requirements

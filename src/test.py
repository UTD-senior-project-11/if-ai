# Read test image
test_image = cv2.imread('test.jpg')
test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_hist = cv2.calcHist([test_gray], [0], None, [256], [0, 256])

# Read data images
data_images = ['data1.jpg', 'data2.jpg']
similarities = []

for data_image_path in data_images:
    # Read data image
    data_image = cv2.imread(data_image_path)
    data_gray = cv2.cvtColor(data_image, cv2.COLOR_BGR2GRAY)
    data_hist = cv2.calcHist([data_gray], [0], None, [256], [0, 256])

    # Compute histogram comparison using Bhattacharyya distance
    similarity = cv2.compareHist(test_hist, data_hist, cv2.HISTCMP_BHATTACHARYYA)
    similarities.append(similarity)

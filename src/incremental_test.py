import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# Define directories for the image dataset
data_dir = 'images/'

# Parameters
image_size = (224, 224)
batch_size = 32

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1. / 255
)

# Load dataset
data = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load MobileNetV2 base model
base_model = MobileNetV2(input_shape=image_size + (3,), include_top=False, weights='imagenet')

# Create feature extraction model
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)


# Extract features from images
def extract_features(data):
    features = []
    labels = []

    for batch in data:
        images, lbls = batch
        features.append(feature_extractor.predict(images))
        labels.append(lbls)

        if len(features) * batch_size >= len(data.filenames):
            break

    features = np.vstack(features)
    labels = np.vstack(labels)

    return features, labels


features, labels = extract_features(data)


# Compute cosine similarity
def compute_similarity(query_feature, features):
    print(query_feature)
    similarities = cosine_similarity(query_feature.reshape(1, -1), features)
    return similarities.flatten()

# Function to find similar images based on cosine similarity
def find_similar_images(query_image_path, threshold=0.8, top_n=5):
    query_image = tf.keras.preprocessing.image.load_img(query_image_path, target_size=image_size)
    query_image = tf.keras.preprocessing.image.img_to_array(query_image)
    query_image = np.expand_dims(query_image, axis=0)
    query_feature = feature_extractor.predict(query_image).flatten()
    print(query_feature)

    similarities = compute_similarity(query_feature, features)

    # Get indices of similar images
    similar_indices = np.where(similarities > threshold)[0]
    similar_indices = similar_indices[similar_indices != data.class_indices[os.path.dirname(query_image_path)]]

    # Sort by similarity and get top n
    sorted_indices = np.argsort(similarities[similar_indices])[::-1][:top_n]

    # Plot similar images
    plt.figure(figsize=(150, 50))

    for i, idx in enumerate(sorted_indices):
        plt.subplot(1, top_n, i + 1)
        img_path = os.path.join(data_dir, data.filenames[similar_indices[idx]])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Similarity: {similarities[similar_indices[idx]]:.2f}")

    plt.show()

# Example usage
query_image_path = 'dog.jpg'
find_similar_images(query_image_path)

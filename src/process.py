import tensorflow as tf

# Load the saved model
loaded_model = tf.keras.models.load_model("model_test.h5")

# Compile the loaded model (if needed)
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Use the loaded model for inference on new images
def predict_image_class(image_path, model):
    from keras.preprocessing import image
    import numpy as np

    # Load and preprocess the image in batches
    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        images.append(img_array)
    
    # stack the images into a single numpy array
    images = np.vstack(images)

    # Perform prediction
    prediction = model.predict(img_array)

    # Get class labels
    class_labels = ["banned", "allowed"]

    # Get predicted class labels for each image
    predicted_classes = [class_labels[int(np.round(pred))] for pred in predictions]

    # Get predicted probabilities for each image
    probabilities = [{"banned": pred[0], "allowed": 1 - pred[0]} for pred in predictions]

    return predicted_classes, probabilities

# Example usage:
# image_path = "cat.jpg"
image_paths = ['cat.jpg, dog.jpg, dog2.jpg']
predicted_classes, probabilities = predict_image_class_batch(image_paths, loaded_model)

for i, image_path in enumerate(image_paths):
    print(f"Image: {image_path}")
    print(f"Predicted Class: {predicted_classes[i]}")
    print("Probabilities:")
    for class_label, prob in probabilities[i].items():
        print(f"{class_label}: {prob:.4f}")
    print()

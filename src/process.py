import tensorflow as tf
import keras.utils as image

# Load the saved model
loaded_model = tf.keras.models.load_model("model_test.h5")

# Compile the loaded model (if needed)
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Use the loaded model for inference on new images
def predict_image_class(image_path, model):
    from keras.preprocessing import image
    import numpy as np

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Perform prediction
    prediction = model.predict(img_array)

    # Determine the predicted class
    if prediction[0][0] >= 0.5:
        return "allowed"
    else:
        return "banned"

# Example usage:
# image_path = "cat.jpg"
image_path = "cat.jpg"
predicted_class = predict_image_class(image_path, loaded_model)
print(f"The image is {predicted_class}.")

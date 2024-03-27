from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
import os

# Load pre-trained model
base_model = ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # 2 classes: banned and allowed

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

img_dir = 'images/'

if not os.path.exists(img_dir):
    raise ValueError(f"Directory '{img_dir}' not found.")

# Use data augmentation to artificially increase the number of training samples
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Create generators for loading images from the directory
train_generator = datagen.flow_from_directory(
    img_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=False,
    seed=42
)

validation_generator = datagen.flow_from_directory(
    img_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=42
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr]
)

# Evaluate
test_generator = datagen.flow_from_directory(
    img_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode='binary',
    shuffle=False,
    seed=42
)

loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Save
model.save("model_test.h5")
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to dataset directories
train_dir = '/mnt/c/Users/Anannya Punia/OneDrive/Desktop/guavaDiseaseDataset/train'
val_dir = '/mnt/c/Users/Anannya Punia/OneDrive/Desktop/guavaDiseaseDataset/val'
test_dir = '/mnt/c/Users/Anannya Punia/OneDrive/Desktop/guavaDiseaseDataset/test'

img_size = (512, 512)
batch_size = 32

# Preprocessing for validation and test sets (no augmentation, only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the images
train_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the CNN model (same as before)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 output classes
])

# Compile the model (no training, just initialization)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save the model directly without training
# Save the model with compression enabled
model.save('guava_disease_model_compressed.keras')


print("Model saved successfully!")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# Define the paths to the train and validation datasets
train_path = "D:/UNI/Internship/Eziline Software House/Bone Fracture/BoneFractureDataset/archive (6)/train"
val_path = "D:/UNI/Internship/Eziline Software House/Bone Fracture/BoneFractureDataset/archive (6)/val"

# Define image size and batch size
img_size = (128, 128)
batch_size = 32

# Create ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load train and validation datasets using flow_from_directory
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Custom callback to stop training when accuracy reaches 80%
class StopTrainingAtAccuracy(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') >= 0.8:
            self.model.stop_training = True

# Train the model with the custom callback
epochs = 100  # Increase the number of epochs to allow the custom callback to trigger
stop_training_callback = StopTrainingAtAccuracy()

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[stop_training_callback]  # Add the custom callback to the list of callbacks
)

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(val_generator, steps=len(val_generator))
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Make predictions on new data
# Example:
# Load and preprocess a new X-ray image
new_image_path = "D:/UNI/Internship/Eziline Software House/Bone Fracture/BoneFractureDataset/archive (6)/val/fractured/1.jpg"
new_image = tf.keras.preprocessing.image.load_img(new_image_path, target_size=img_size)
new_image = tf.keras.preprocessing.image.img_to_array(new_image)
new_image = np.expand_dims(new_image, axis=0)
new_image = new_image / 255.0  # Normalize the pixel values to [0, 1]

# Make predictions
prediction = model.predict(new_image)
if prediction[0][0] >= 0.5:
    print("The X-ray image contains a fracture.")
else:
    print("The X-ray image does not contain a fracture.")
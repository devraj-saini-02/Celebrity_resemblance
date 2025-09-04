import kagglehub
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import shutil

# --------------------------------------------------------------------
# Download dataset
# --------------------------------------------------------------------
path = kagglehub.dataset_download("vishesh1412/celebrity-face-image-dataset")
dataset_dir = f"{path}/Celebrity Faces Dataset"
print("Dataset downloaded:", dataset_dir)

# --------------------------------------------------------------------
# Training/Validation datasets
# --------------------------------------------------------------------
train_ds = keras.utils.image_dataset_from_directory(
    dataset_dir, labels='inferred', label_mode='int',
    batch_size=32, image_size=(224,224),
    validation_split=0.2, subset='training', seed=123
)

val_ds = keras.utils.image_dataset_from_directory(
    dataset_dir, labels='inferred', label_mode='int',
    batch_size=32, image_size=(224,224),
    validation_split=0.2, subset='validation', seed=123
)

# Save class names (for later inference)
class_names = train_ds.class_names
with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# --------------------------------------------------------------------
# Data Augmentation
# --------------------------------------------------------------------
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])
train_ds = train_ds.map(lambda x,y: (data_augmentation(x, training=True), y))

# --------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x,y: (preprocess_input(tf.cast(x, tf.float32)), y)).cache().prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x,y: (preprocess_input(tf.cast(x, tf.float32)), y)).cache().prefetch(AUTOTUNE)

# --------------------------------------------------------------------
# CNN MODEL (Transfer Learning - ResNet50)
# --------------------------------------------------------------------
conv_base = keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))
for layer in conv_base.layers:
    if "conv4_block" in layer.name or "conv5_block" in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

model = Sequential([
    conv_base,
    layers.GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(len(class_names), activation="softmax")
])

model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# --------------------------------------------------------------------
# Training
# --------------------------------------------------------------------
checkpoint_cb = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
history = model.fit(train_ds, epochs=6, validation_data=val_ds, callbacks=[checkpoint_cb])

# --------------------------------------------------------------------
# Save sample celebrity images for backend
# --------------------------------------------------------------------
output_dir = "sample_images"
os.makedirs(output_dir, exist_ok=True)

for name in class_names:
    celeb_dir = os.path.join(dataset_dir, name)
    first_img = os.listdir(celeb_dir)[0]  # take first image
    src = os.path.join(celeb_dir, first_img)
    dst = os.path.join(output_dir, f"{name}.jpg")
    shutil.copy(src, dst)

print("Training complete.")
print("Model saved as best_model.keras")
print("Class names saved in class_names.txt")
print("Sample celebrity images saved in sample_images/")


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential 
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
import gradio as gr

#Generators
train_ds = keras.utils.image_dataset_from_directory(
    directory='/...../celebrity-face-image-dataset/Celebrity Faces Dataset',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224,224),  
    validation_split=0.2,
    subset='training',
    seed=123
)
validation_ds = keras.utils.image_dataset_from_directory(
    directory='/...../celebrity-face-image-dataset/Celebrity Faces Dataset',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224,224),
    validation_split=0.2,
    subset='validation',
    seed=123
)

# Data Augmentation
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

#Preprocessing
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)).cache().prefetch(AUTOTUNE)
validation_ds   = validation_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)).cache().prefetch(AUTOTUNE)

#CNN MODEL (Finetuned - Validation Accuracy:82.5%)
    #Tranfer Learing 
conv_base=keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
set_trainable = False
for layer in conv_base.layers:
     if ("conv4_block" in layer.name) or ("conv5_block" in layer.name) or ("conv3_block" in layer.name):
        layer.trainable = True
    else:
        layer.trainable = False

    #Architecture
model=Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
num_classes = 17
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=keras.optimizers.Adam(1e-4),loss='sparse_categorical_crossentropy',  metrics=['accuracy'])

#Training 
checkpoint_cb = ModelCheckpoint("best_model.keras", monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
history = model.fit(train_ds, epochs=50, validation_data=validation_ds, callbacks=[checkpoint_cb])
best_model = keras.models.load_model("best_model.keras")

#Prediction Function
temp_ds = keras.utils.image_dataset_from_directory(
    directory='/...../celebrity-face-image-dataset/Celebrity Faces Dataset',
    labels='inferred',
    label_mode='int',
    image_size=(224,224),
    shuffle=False 
)
class_names = temp_ds.class_names
del temp_ds
def predict_and_show(img):
    img_resized = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized), axis=0)
    img_array = preprocess_input(img_array)
    preds = best_model.predict(img_array)
    label = class_names[np.argmax(preds[0])]
    return label

#Launching Model 
demo = gr.Interface(
    fn=predict_and_show,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Which celebrity are you?"
)

demo.launch()


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

app = FastAPI()

# Load trained model
best_model = keras.models.load_model("best_model.keras")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f]

# Directory of saved sample images
sample_dir = "sample_images"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Take uploaded image → return first saved image of predicted celebrity"""
    img = Image.open(file.file).convert("RGB").resize((224,224))
    arr = np.expand_dims(np.array(img), axis=0)
    arr = preprocess_input(arr)

    preds = best_model.predict(arr)
    label = class_names[np.argmax(preds[0])]

    celeb_image = os.path.join(sample_dir, f"{label}.jpg")
    return FileResponse(celeb_image)

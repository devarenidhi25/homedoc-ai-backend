import tensorflow as tf
import json
import numpy as np
from utils.image_preprocessing import preprocess_image

# Load model and labels ONCE
MODEL_PATH = "model/image_model/pneumonia_model.h5"
CLASS_NAMES_PATH = "model/image_model/class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)  # {0: "NORMAL", 1: "PNEUMONIA"}


def predict_image(image_bytes: bytes):
    """
    Run inference on uploaded medical image.
    """
    processed_image = preprocess_image(image_bytes)

    prediction = model.predict(processed_image)[0][0]

    confidence = float(prediction)
    label_index = 1 if prediction >= 0.5 else 0
    label = class_names[str(label_index)]

    return {
        "prediction": label,
        "confidence": round(confidence if label_index == 1 else 1 - confidence, 3),
        "disclaimer": "This is an AI-assisted result and not a medical diagnosis."
    }

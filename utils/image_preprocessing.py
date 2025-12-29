import numpy as np
import cv2

IMAGE_SIZE = (224, 224)

def preprocess_image(image_bytes: bytes):
    """
    Convert raw image bytes into a normalized tensor
    suitable for CNN inference.
    """
    # Convert bytes to NumPy array
    np_arr = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image")

    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, IMAGE_SIZE)

    # Normalize
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

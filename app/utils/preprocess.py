import numpy as np

def preprocess_image(image):
    return image.astype(np.float32) / 255.0

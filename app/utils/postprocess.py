import numpy as np

def postprocess_image(image):
    return (image * 255).astype(np.uint8)

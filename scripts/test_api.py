import requests
import numpy as np
import matplotlib.pyplot as plt
import os

url = "http://127.0.0.1:5004/image/reconstruct"

input_image_path = "./app/static/input_images/original_image.png"
output_dir = "./app/static/output_images"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"L'image d'entrée est introuvable : {input_image_path}")

with open(input_image_path, "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    reconstructed_img = np.array(data["reconstructed"])


    output_image_path = os.path.join(output_dir, "reconstructed_image.png")
    plt.imsave(output_image_path, reconstructed_img, cmap="gray")
    print(f"Image reconstruite sauvegardée dans : {output_image_path}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    original_img = plt.imread(input_image_path)
    axes[0].imshow(original_img, cmap="gray")
    axes[0].set_title("Image originale")
    axes[0].axis("off")
    axes[1].imshow(reconstructed_img, cmap="gray")
    axes[1].set_title("Image reconstruite")
    axes[1].axis("off")
    plt.show()
else:
    print(f"Erreur {response.status_code} : {response.text}")

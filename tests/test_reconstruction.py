import torch
from models.vae import VAE
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

model = VAE()
model_path = "./models/vae_mnist.pth"

if not torch.load(model_path):
    raise FileNotFoundError(f"Modèle non trouvé à : {model_path}")

model.load_state_dict(torch.load(model_path))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

sample_img, _ = test_dataset[0]
sample_img = sample_img.unsqueeze(0) 


with torch.no_grad():
    reconstructed_img, _, _ = model(sample_img)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(sample_img.squeeze().numpy(), cmap="gray")
axes[0].set_title("Image originale")
axes[0].axis("off")
axes[1].imshow(reconstructed_img.squeeze().numpy(), cmap="gray")
axes[1].set_title("Image reconstruite")
axes[1].axis("off")
plt.show()

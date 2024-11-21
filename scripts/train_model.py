import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vae import VAE
import os

batch_size = 64
epochs = 10
learning_rate = 1e-3

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = VAE()
criterion = nn.BCELoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss / len(train_loader.dataset)}")

model_path = "./models/vae_mnist.pth"
torch.save(model.state_dict(), model_path)
print(f"Modèle sauvegardé à : {model_path}")

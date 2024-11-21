import torch
from models.vae import VAE
import os

model = VAE()
model_path = "./models/vae_mnist.pth"
onnx_path = "./public/vae_mnist.onnx"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le modèle n'existe pas à : {model_path}")

model.load_state_dict(torch.load(model_path))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"Modèle exporté en ONNX à : {onnx_path}")

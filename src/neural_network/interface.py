import torch
from src.neural_network.srcnn_model import SRCNN

# RGB mean/std za denormalizaciju
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

def apply_srcnn(image_tensor, model_path, device="cpu"):
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        # Normalizuj pre inference
        x = (image_tensor - mean[:, None, None]) / std[:, None, None]
        output = model(x.unsqueeze(0)).squeeze(0)
        # Denormalizacija
        output = output * std[:, None, None] + mean[:, None, None]
        output = output.clamp(0.0, 1.0)
    return output

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.neural_network.dataset import SRCNNDataset
from src.neural_network.srcnn_model import SRCNN


def psnr(pred, target):
    mse = nn.functional.mse_loss(pred, target)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = SRCNN().to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    dataset = SRCNNDataset(
        lr_dir=Path("data/train/degraded"),
        hr_dir=Path("data/train/original"),
        scale=4,
        patch=128
    )

    dataset = Subset(dataset, range(min(300, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    epochs = 12
    for e in range(epochs):
        loss_sum = 0
        psnr_sum = 0

        for lr, hr in loader:
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()
            out = model(lr)
            out = torch.clamp(out, 0, 1)

            loss = criterion(out, hr)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            psnr_sum += psnr(out, hr).item()

        print(
            f"Epoch [{e+1}/{epochs}] "
            f"RMSE: {(loss_sum/len(loader))**0.5:.4f} "
            f"PSNR: {psnr_sum/len(loader):.2f} dB"
        )

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/srcnn_y.pth")
    print("Model saved: models/srcnn_y.pth")


if __name__ == "__main__":
    train()

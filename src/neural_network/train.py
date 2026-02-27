import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path

from src.neural_network.dataset import SRCNNDataset
from src.neural_network.srcnn_model import SRCNN


def psnr(pred, target):
    mse = nn.functional.mse_loss(pred, target)
    mse = torch.clamp(mse, min=1e-12)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Train Y-channel SRCNN model")
    parser.add_argument("--lr-dir", type=Path, default=Path("data/train/degraded"), help="LR train folder")
    parser.add_argument("--hr-dir", type=Path, default=Path("data/train/original"), help="HR train folder")
    parser.add_argument("--scale", type=int, default=3, help="Upscale factor")
    parser.add_argument("--patch", type=int, default=33, help="LR patch size")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 regularization")
    parser.add_argument("--max-samples", type=int, default=800, help="Maximum images to use")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=1e-5, help="Minimum val loss improvement")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = SRCNN().to(device)

    # Diferencijalni learning rate: conv3 (output) uci sporije (kao u originalnom SRCNN radu)
    optimizer = optim.Adam([
        {'params': model.conv1.parameters(), 'lr': args.learning_rate},
        {'params': model.conv2.parameters(), 'lr': args.learning_rate},
        {'params': model.conv3.parameters(), 'lr': args.learning_rate * 0.1},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    dataset = SRCNNDataset(
        lr_dir=args.lr_dir,
        hr_dir=args.hr_dir,
        scale=args.scale,
        patch=args.patch,
        augment=False,
        samples_multiplier=1,
    )

    # Split by image index (pre-multiplier)
    num_images = len(dataset.image_pairs)
    total_images = min(args.max_samples, num_images)
    if total_images == 0:
        raise ValueError(f"Dataset je prazan. Proveri {args.lr_dir} i {args.hr_dir}.")

    all_indices = torch.randperm(num_images, generator=torch.Generator().manual_seed(args.seed)).tolist()
    selected = all_indices[:total_images]

    val_count = max(1, int(args.val_ratio * total_images))
    train_count = total_images - val_count

    train_img_indices = selected[:train_count]
    val_img_indices = selected[train_count:]

    # Train dataset sa augmentacijom i 10 random patch-eva po slici po epohi
    MULTIPLIER = 10
    train_dataset = SRCNNDataset.__new__(SRCNNDataset)
    train_dataset.__dict__.update(dataset.__dict__)
    train_dataset.augment = True
    train_dataset.samples_multiplier = MULTIPLIER

    # Expand image indices to patch indices
    train_indices = []
    for img_i in train_img_indices:
        for m in range(MULTIPLIER):
            train_indices.append(img_i + m * num_images)

    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(dataset, val_img_indices)

    print(f"Train: {len(train_set)} patches ({train_count} images x {MULTIPLIER}) | Val: {len(val_set)} images")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for e in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_psnr_sum = 0.0

        for lr, hr in train_loader:
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()
            out = model(lr)

            loss = criterion(out, hr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_psnr_sum += psnr(torch.clamp(out, 0, 1), hr).item()

        model.eval()
        val_loss_sum = 0.0
        val_psnr_sum = 0.0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr = lr.to(device)
                hr = hr.to(device)
                out = model(lr)
                val_loss = criterion(out, hr)

                val_loss_sum += val_loss.item()
                val_psnr_sum += psnr(torch.clamp(out, 0, 1), hr).item()

        avg_train_loss = train_loss_sum / max(1, len(train_loader))
        avg_train_psnr = train_psnr_sum / max(1, len(train_loader))
        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        avg_val_psnr = val_psnr_sum / max(1, len(val_loader))

        print(
            f"Epoch [{e+1}/{args.epochs}] "
            f"Train RMSE: {avg_train_loss**0.5:.4f} "
            f"Train PSNR: {avg_train_psnr:.2f} dB | "
            f"Val RMSE: {avg_val_loss**0.5:.4f} "
            f"Val PSNR: {avg_val_psnr:.2f} dB"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < (best_val_loss - args.min_delta):
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "models/srcnn_y_best.pth")
            print("Saved best model: models/srcnn_y_best.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping activated at epoch {e+1} (patience={args.patience}).")
                break

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/srcnn_y.pth")
    print("Model saved: models/srcnn_y.pth")


if __name__ == "__main__":
    parser = _build_arg_parser()
    train(parser.parse_args())

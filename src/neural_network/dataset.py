import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image

from src.utils.image_io import load_image, rgb_to_ycbcr


class SRCNNDataset(Dataset):
    def __init__(self, lr_dir: Path, hr_dir: Path, scale=4, patch=128):
        self.lr = sorted(lr_dir.glob("*"))
        self.hr = sorted(hr_dir.glob("*"))
        self.scale = scale
        self.patch = patch

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        lr = load_image(self.lr[idx])
        hr = load_image(self.hr[idx])

        y_lr, _, _ = rgb_to_ycbcr(lr)
        y_hr, _, _ = rgb_to_ycbcr(hr)

        y_lr = y_lr / 255.0
        y_hr = y_hr / 255.0

        h, w = y_lr.shape
        ph = min(self.patch, h)
        pw = min(self.patch, w)

        top = np.random.randint(0, h - ph + 1)
        left = np.random.randint(0, w - pw + 1)

        lr_patch = y_lr[top:top+ph, left:left+pw]

        hr_patch = y_hr[
            top*self.scale:(top+ph)*self.scale,
            left*self.scale:(left+pw)*self.scale
        ]

        # bicubic upscale LR patch
        lr_pil = Image.fromarray((lr_patch * 255).astype(np.uint8))
        lr_up = lr_pil.resize(
            (pw * self.scale, ph * self.scale),
            Image.BICUBIC
        )
        lr_up = np.array(lr_up) / 255.0

        lr_up = torch.from_numpy(lr_up).unsqueeze(0).float()
        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0).float()

        return lr_up, hr_patch

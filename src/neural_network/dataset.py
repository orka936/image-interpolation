import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

from src.utils.image_io import load_image, rgb_to_ycbcr
from src.interpolation.bicubic import bicubic_interpolation


class SRCNNDataset(Dataset):
    """Dataset koji za svaku sliku kešira Y kanal i upscale rezultat.

    Keširanjem se izbegava ponavljanje skupog I/O i bicubic interpolacije
    tokom treninga. Svaki __getitem__ izvlači random patch iz keširanih slika.
    """

    def __init__(self, lr_dir: Path, hr_dir: Path, scale=3, patch=33,
                 augment=False, samples_multiplier=1):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.patch = patch
        self.augment = augment
        self.samples_multiplier = samples_multiplier

        self.image_pairs = self._build_pairs()
        if not self.image_pairs:
            raise ValueError("Nema validnih LR/HR parova u zadatim direktorijumima.")

        # Pre-compute i kesiraj sve Y kanale + upscale rezultate
        self._y_lr_up = []   # list of upscaled Y (0-1), shape (H*scale, W*scale)
        self._y_hr = []      # list of HR Y (0-1), shape (H*scale, W*scale)
        print(f"Keširam {len(self.image_pairs)} slika...", end="", flush=True)
        for i, (lr_path, hr_path) in enumerate(self.image_pairs):
            lr = load_image(lr_path)
            hr = load_image(hr_path)

            y_lr, _, _ = rgb_to_ycbcr(lr)
            y_hr, _, _ = rgb_to_ycbcr(hr)

            y_lr = y_lr / 255.0
            y_hr = y_hr / 255.0

            h, w = y_lr.shape
            exp_h = h * self.scale
            exp_w = w * self.scale

            # Crop HR to exact expected size
            y_hr = y_hr[:exp_h, :exp_w]

            # Upscale LR jednom (umesto svaki put u __getitem__)
            y_lr_up = bicubic_interpolation(y_lr, scale_factor=self.scale)

            self._y_lr_up.append(y_lr_up.astype(np.float32))
            self._y_hr.append(y_hr.astype(np.float32))

            if (i + 1) % 100 == 0:
                print(f" {i+1}", end="", flush=True)
        print(" Done.")

    def __len__(self):
        return len(self.image_pairs) * self.samples_multiplier

    def _build_pairs(self):
        valid_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        lr_files = {
            path.stem: path
            for path in self.lr_dir.glob("*")
            if path.is_file() and path.suffix.lower() in valid_ext
        }
        hr_files = {
            path.stem: path
            for path in self.hr_dir.glob("*")
            if path.is_file() and path.suffix.lower() in valid_ext
        }
        common_keys = sorted(set(lr_files.keys()) & set(hr_files.keys()))
        return [(lr_files[key], hr_files[key]) for key in common_keys]

    def __getitem__(self, idx):
        img_idx = idx % len(self.image_pairs)
        y_lr_up = self._y_lr_up[img_idx]
        y_hr = self._y_hr[img_idx]

        # HR dimenzije (vec upscaled size)
        hr_h, hr_w = y_hr.shape
        ph = min(self.patch * self.scale, hr_h)
        pw = min(self.patch * self.scale, hr_w)

        top = np.random.randint(0, hr_h - ph + 1)
        left = np.random.randint(0, hr_w - pw + 1)

        lr_patch = y_lr_up[top:top+ph, left:left+pw]
        hr_patch = y_hr[top:top+ph, left:left+pw]

        if self.augment:
            lr_patch, hr_patch = self._augment_pair(lr_patch, hr_patch)

        lr_t = torch.from_numpy(np.ascontiguousarray(lr_patch)).unsqueeze(0).float()
        hr_t = torch.from_numpy(np.ascontiguousarray(hr_patch)).unsqueeze(0).float()

        return lr_t, hr_t

    def _augment_pair(self, a, b):
        if np.random.rand() < 0.5:
            a = np.flip(a, axis=1)
            b = np.flip(b, axis=1)
        if np.random.rand() < 0.5:
            a = np.flip(a, axis=0)
            b = np.flip(b, axis=0)
        k = np.random.randint(0, 4)
        if k > 0:
            a = np.rot90(a, k=k)
            b = np.rot90(b, k=k)
        return a.copy(), b.copy()

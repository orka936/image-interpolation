import numpy as np
from PIL import Image
from pathlib import Path


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(img: np.ndarray, path: Path):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def rgb_to_ycbcr(img: np.ndarray):
    img = img.astype(np.float32)

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr):
    y  = y.astype(np.float32)
    cb = cb.astype(np.float32) - 128
    cr = cr.astype(np.float32) - 128

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    img = np.stack([r, g, b], axis=2)
    return np.clip(img, 0, 255)

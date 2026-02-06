# Pillow učitavanje/čuvanje
from PIL import Image
import numpy as np


def load_image(path):
    """
    Učitava sliku sa diska i vraća je kao NumPy niz tipa float32.
    """
    img = Image.open(path).convert("RGB")
    img_np = np.asarray(img, dtype=np.float32)
    return img_np


def save_image(image_np, path):
    """
    Čuva NumPy niz kao RGB sliku.
    Pretpostavlja da su vrednosti u opsegu [0, 255].
    """
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    img = Image.fromarray(image_np, mode="RGB")
    img.save(path)
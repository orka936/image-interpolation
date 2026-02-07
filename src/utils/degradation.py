# downsampling, maskiranje piksela
import numpy as np
from PIL import Image

def downsample_image(image: np.ndarray, scale_factor: int) -> np.ndarray:
    """Downsample image with bicubic interpolation."""
    img_pil = Image.fromarray(np.uint8(image))
    new_w = img_pil.width // scale_factor
    new_h = img_pil.height // scale_factor
    img_small = img_pil.resize((new_w, new_h), resample=Image.BICUBIC)
    return np.array(img_small)

def remove_pixels(image, missing_ratio):
    corrupted = image.copy()
    h, w, c = corrupted.shape

    num_pixels = h * w
    num_missing = int(num_pixels * missing_ratio)

    indices = np.random.choice(num_pixels, num_missing, replace=False)

    for idx in indices:
        y = idx // w
        x = idx % w
        corrupted[y, x, :] = 0

    return corrupted

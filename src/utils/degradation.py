# downsampling, maskiranje piksela
import numpy as np


def downsample_image(image, scale_factor):
    h, w, c = image.shape
    new_h = h // scale_factor
    new_w = w // scale_factor

    downsampled = image[0:new_h * scale_factor:scale_factor,
                         0:new_w * scale_factor:scale_factor,
                         :]
    return downsampled


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

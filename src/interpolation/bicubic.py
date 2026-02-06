import numpy as np
from scipy.interpolate import RegularGridInterpolator

def bicubic_interpolation(image, scale_factor=2, missing_mask=None):
    """
    OPTIMIZOVANA bikubna interpolacija.
    Koristi scipy RegularGridInterpolator za brzinu.
    """
    if missing_mask is None:
        return bicubic_upscale(image, scale_factor)
    else:
        return bicubic_reconstruct(image, missing_mask)

def bicubic_upscale(image, scale_factor):
    """Bikubna interpolacija za upscaling koristeći scipy."""
    if len(image.shape) == 3:
        h, w, c = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        result = np.zeros((new_h, new_w, c), dtype=np.float32)
        for ch in range(c):
            result[:, :, ch] = bicubic_upscale_channel(image[:, :, ch], scale_factor)
        return result
    else:
        return bicubic_upscale_channel(image, scale_factor)

def bicubic_upscale_channel(channel, scale_factor):
    """Bikubna interpolacija za jedan kanal."""
    h, w = channel.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Kreiraj interpolator
    x = np.arange(w)
    y = np.arange(h)
    interp = RegularGridInterpolator((y, x), channel, method='cubic', bounds_error=False, fill_value=0)
    
    # Kreiraj nove koordinate
    x_new = np.linspace(0, w-1, new_w)
    y_new = np.linspace(0, h-1, new_h)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    points = np.stack([Y_new.ravel(), X_new.ravel()], axis=1)
    
    # Interpolacija
    result = interp(points).reshape(new_h, new_w)
    return result

def bicubic_reconstruct(image, missing_mask):
    """Rekonstrukcija nedostajućih piksela."""
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for ch in range(image.shape[2]):
            result[:, :, ch] = bicubic_reconstruct_channel(image[:, :, ch], missing_mask)
        return result
    else:
        return bicubic_reconstruct_channel(image, missing_mask)

def bicubic_reconstruct_channel(channel, missing_mask):
    """Rekonstrukcija koristeći konvoluciju sa većom okolinom."""
    from scipy.ndimage import convolve
    
    output = channel.copy()
    h, w = channel.shape
    
    # Kreiraj kernel za 5x5 okolinu
    kernel = np.ones((5, 5), dtype=np.float32)
    kernel[2, 2] = 0  # centralni piksel
    
    # Izračunaj sumu validnih suseda
    valid_mask = (missing_mask == 0).astype(np.float32)
    neighbor_sum = convolve(channel * valid_mask, kernel, mode='constant')
    neighbor_count = convolve(valid_mask, kernel, mode='constant')
    
    # Zameni nedostajuće piksele
    replace_mask = (missing_mask == 1) & (neighbor_count > 0)
    output[replace_mask] = neighbor_sum[replace_mask] / neighbor_count[replace_mask]
    
    return output
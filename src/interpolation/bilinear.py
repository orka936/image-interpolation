import numpy as np

def bilinear_interpolation(image, scale_factor=2, missing_mask=None):
    """
    OPTIMIZOVANA bilinearna interpolacija.
    """
    if missing_mask is None:
        return bilinear_upscale(image, scale_factor)
    else:
        return bilinear_reconstruct(image, missing_mask)

def bilinear_upscale(image, scale_factor):
    """Vektorizovana bilinearna interpolacija za upscaling."""
    if len(image.shape) == 3:
        h, w, c = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        result = np.zeros((new_h, new_w, c), dtype=np.float32)
        for ch in range(c):
            result[:, :, ch] = bilinear_upscale_channel(image[:, :, ch], scale_factor)
        return result
    else:
        return bilinear_upscale_channel(image, scale_factor)

def bilinear_upscale_channel(channel, scale_factor):
    """Vektorizovana bilinearna interpolacija za jedan kanal."""
    h, w = channel.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Kreiraj mape koordinata
    x = np.linspace(0, w-1, new_w)
    y = np.linspace(0, h-1, new_h)
    
    # Indeksi susednih piksela
    x0 = np.floor(x).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    
    # Težine
    wx = x - x0
    wy = y - y0
    
    # Ekstraktuj vrednosti suseda (broadcast-friendly)
    q00 = channel[y0[:, None], x0]
    q10 = channel[y1[:, None], x0]
    q01 = channel[y0[:, None], x1]
    q11 = channel[y1[:, None], x1]
    
    # Interpolacija
    result = (1 - wy[:, None]) * (1 - wx) * q00 + \
             wy[:, None] * (1 - wx) * q10 + \
             (1 - wy[:, None]) * wx * q01 + \
             wy[:, None] * wx * q11
    
    return result

def bilinear_reconstruct(image, missing_mask):
    """Optimizovana rekonstrukcija nedostajućih piksela."""
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for ch in range(image.shape[2]):
            result[:, :, ch] = bilinear_reconstruct_channel(image[:, :, ch], missing_mask)
        return result
    else:
        return bilinear_reconstruct_channel(image, missing_mask)

def bilinear_reconstruct_channel(channel, missing_mask):
    """Rekonstrukcija koristeći konvoluciju za brzinu."""
    from scipy.ndimage import convolve
    
    output = channel.copy()
    h, w = channel.shape
    
    # Kreiraj kernel za prosečnu vrednost suseda (3x3)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.float32)
    
    # Izračunaj sumu validnih suseda za svaki piksel
    valid_mask = (missing_mask == 0).astype(np.float32)
    neighbor_sum = convolve(channel * valid_mask, kernel, mode='constant')
    neighbor_count = convolve(valid_mask, kernel, mode='constant')
    
    # Zameni nedostajuće piksele prosekom suseda
    replace_mask = (missing_mask == 1) & (neighbor_count > 0)
    output[replace_mask] = neighbor_sum[replace_mask] / neighbor_count[replace_mask]
    
    return output
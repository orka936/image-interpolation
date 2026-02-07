import numpy as np

def bilinear_interpolation(image, scale_factor=4, missing_mask=None):
    """
    Optimizovana bilinearna interpolacija sa vektorizacijom.
    """
    if missing_mask is None:
        return bilinear_upscale_vectorized(image, scale_factor)
    else:
        return bilinear_reconstruct_vectorized(image, missing_mask)

def bilinear_upscale_vectorized(image, scale_factor):
    """Potpuno vektorizovana bilinearna interpolacija."""
    if len(image.shape) == 3:
        h, w, c = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Kreiraj mrežu koordinata
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
        
        # 3D interpolacija - obrađujemo sve kanale odjednom
        result = np.zeros((new_h, new_w, c), dtype=np.float32)
        
        for ch in range(c):
            channel = image[:, :, ch]
            
            # Vrednosti 4 suseda za SVE piksele odjednom
            q00 = channel[y0[:, None], x0]
            q10 = channel[y1[:, None], x0]
            q01 = channel[y0[:, None], x1]
            q11 = channel[y1[:, None], x1]
            
            # Interpolacija koristeći broadcasting
            result[:, :, ch] = (1 - wy[:, None]) * (1 - wx) * q00 + \
                               wy[:, None] * (1 - wx) * q10 + \
                               (1 - wy[:, None]) * wx * q01 + \
                               wy[:, None] * wx * q11
        
        return result
    else:
        # Za jednokanalne slike
        h, w = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        x = np.linspace(0, w-1, new_w)
        y = np.linspace(0, h-1, new_h)
        
        x0 = np.floor(x).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(y).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)
        
        wx = x - x0
        wy = y - y0
        
        q00 = image[y0[:, None], x0]
        q10 = image[y1[:, None], x0]
        q01 = image[y0[:, None], x1]
        q11 = image[y1[:, None], x1]
        
        return (1 - wy[:, None]) * (1 - wx) * q00 + \
               wy[:, None] * (1 - wx) * q10 + \
               (1 - wy[:, None]) * wx * q01 + \
               wy[:, None] * wx * q11

def bilinear_reconstruct_vectorized(image, missing_mask):
    """Vektorizovana rekonstrukcija."""
    if len(image.shape) == 3:
        h, w, c = image.shape
        result = np.zeros_like(image)
        
        for ch in range(c):
            result[:, :, ch] = bilinear_reconstruct_channel_vectorized(image[:, :, ch], missing_mask)
        
        return result
    else:
        return bilinear_reconstruct_channel_vectorized(image, missing_mask)

def bilinear_reconstruct_channel_vectorized(channel, missing_mask):
    """Rekonstrukcija koristeći vektorizovano filtriranje."""
    h, w = channel.shape
    
    # Kreiraj kopiju
    output = channel.copy()
    
    # Pronađi sve nedostajuće piksele
    missing_y, missing_x = np.where(missing_mask)
    
    if len(missing_y) == 0:
        return output
    
    # Za svaki nedostajući piksel, izračunaj prosek suseda
    for idx in range(len(missing_y)):
        i, j = missing_y[idx], missing_x[idx]
        
        # Definiši okolinu 3x3
        i_min, i_max = max(0, i-1), min(h, i+2)
        j_min, j_max = max(0, j-1), min(w, j+2)
        
        # Izvuci okolinu
        neighborhood = channel[i_min:i_max, j_min:j_max]
        neighborhood_mask = missing_mask[i_min:i_max, j_min:j_max]
        
        # Ravnaj za indeksiranje
        neighborhood_flat = neighborhood.ravel()
        neighborhood_mask_flat = neighborhood_mask.ravel()
        
        # Ukloni nedostajuće piksele iz okoline
        valid_values = neighborhood_flat[~neighborhood_mask_flat]
        
        if len(valid_values) > 0:
            output[i, j] = np.mean(valid_values)
    
    return output
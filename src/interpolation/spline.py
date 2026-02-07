import numpy as np

def spline_interpolation(image, scale_factor=4, missing_mask=None):
    """
    Optimizovana spline interpolacija.
    """
    if missing_mask is None:
        return spline_upscale_fast(image, scale_factor)
    else:
        return spline_reconstruct_fast(image, missing_mask)

def spline_upscale_fast(image, scale_factor):
    """Brza spline interpolacija koristeći separabilnost."""
    if len(image.shape) == 3:
        h, w, c = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        result = np.zeros((new_h, new_w, c), dtype=np.float32)
        
        # Prvo uvećaj po redovima, pa po kolonama
        for ch in range(c):
            channel = image[:, :, ch]
            
            # 1. Interpolacija po redovima
            temp = np.zeros((h, new_w), dtype=np.float32)
            
            # Kreiraj koordinate
            x_old = np.arange(w)
            x_new = np.linspace(0, w-1, new_w)
            
            for row in range(h):
                # Koristi NumPy interpolaciju za brzinu
                temp[row, :] = np.interp(x_new, x_old, channel[row, :])
            
            # 2. Interpolacija po kolonama
            y_old = np.arange(h)
            y_new = np.linspace(0, h-1, new_h)
            
            for col in range(new_w):
                result[:, col, ch] = np.interp(y_new, y_old, temp[:, col])
        
        return result
    else:
        # Za jednokanalne slike
        h, w = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Interpolacija po redovima
        temp = np.zeros((h, new_w), dtype=np.float32)
        x_old = np.arange(w)
        x_new = np.linspace(0, w-1, new_w)
        
        for row in range(h):
            temp[row, :] = np.interp(x_new, x_old, image[row, :])
        
        # Interpolacija po kolonama
        result = np.zeros((new_h, new_w), dtype=np.float32)
        y_old = np.arange(h)
        y_new = np.linspace(0, h-1, new_h)
        
        for col in range(new_w):
            result[:, col] = np.interp(y_new, y_old, temp[:, col])
        
        return result

def spline_reconstruct_fast(image, missing_mask):
    """Brza rekonstrukcija spline-om."""
    if len(image.shape) == 3:
        h, w, c = image.shape
        result = np.zeros_like(image)
        
        for ch in range(c):
            result[:, :, ch] = spline_reconstruct_channel_fast(image[:, :, ch], missing_mask)
        
        return result
    else:
        return spline_reconstruct_channel_fast(image, missing_mask)

def spline_reconstruct_channel_fast(channel, missing_mask):
    """Rekonstrukcija koristeći iterativno popunjavanje."""
    h, w = channel.shape
    output = channel.copy()
    
    # Kreiraj kopiju maske za praćenje
    remaining_mask = missing_mask.copy()
    
    # Broj nedostajućih piksela
    remaining_count = np.sum(remaining_mask)
    
    # Iterativno popunjavanje
    iteration = 0
    max_iterations = 5
    
    while remaining_count > 0 and iteration < max_iterations:
        iteration += 1
        
        # Pronađi nedostajuće piksele koji imaju najmanje 3 suseda
        missing_y, missing_x = np.where(remaining_mask)
        filled_count = 0
        
        for idx in range(len(missing_y)):
            i, j = missing_y[idx], missing_x[idx]
            
            # Proveri 3x3 okolinu
            i_min, i_max = max(0, i-1), min(h, i+2)
            j_min, j_max = max(0, j-1), min(w, j+2)
            
            neighborhood = output[i_min:i_max, j_min:j_max]
            neighborhood_mask = remaining_mask[i_min:i_max, j_min:j_max]
            
            # Ravnaj za indeksiranje
            neighborhood_flat = neighborhood.ravel()
            neighborhood_mask_flat = neighborhood_mask.ravel()
            
            # Broj validnih suseda
            valid_count = np.sum(~neighborhood_mask_flat)
            
            if valid_count >= 3:
                # Izračunaj vrednost na osnovu suseda
                valid_values = neighborhood_flat[~neighborhood_mask_flat]
                output[i, j] = np.mean(valid_values)
                remaining_mask[i, j] = False
                filled_count += 1
        
        # Ako nismo popunili nijedan piksel, izađi
        if filled_count == 0:
            break
        
        remaining_count = np.sum(remaining_mask)
    
    # Za preostale piksele, koristi jednostavniji pristup
    if remaining_count > 0:
        missing_y, missing_x = np.where(remaining_mask)
        
        for idx in range(len(missing_y)):
            i, j = missing_y[idx], missing_x[idx]
            
            # Proširi okolinu ako je potrebno
            i_min, i_max = max(0, i-2), min(h, i+3)
            j_min, j_max = max(0, j-2), min(w, j+3)
            
            neighborhood = output[i_min:i_max, j_min:j_max]
            neighborhood_mask = remaining_mask[i_min:i_max, j_min:j_max]
            
            # Ravnaj za indeksiranje
            neighborhood_flat = neighborhood.ravel()
            neighborhood_mask_flat = neighborhood_mask.ravel()
            
            valid_values = neighborhood_flat[~neighborhood_mask_flat]
            
            if len(valid_values) > 0:
                output[i, j] = np.mean(valid_values)
    
    return output
import numpy as np

def cubic_interpolate(p, x):
    """
    Catmull-Rom kubna interpolacija za 1D niz od 4 tačke.
    """
    return p[1] + 0.5 * x * (p[2] - p[0] + 
                            x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + 
                                x * (3.0 * p[1] - p[0] - 3.0 * p[2] + p[3])))

def bicubic_interpolation(image, scale_factor=4, missing_mask=None):
    """
    Optimizovana bikubna interpolacija sa delimičnom vektorizacijom.
    """
    if missing_mask is None:
        return bicubic_upscale_optimized(image, scale_factor)
    else:
        return bicubic_reconstruct_optimized(image, missing_mask)

def bicubic_upscale_optimized(image, scale_factor):
    """Optimizovana bikubna interpolacija."""
    if len(image.shape) == 3:
        h, w, c = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Kreiraj mrežu koordinata
        x_new = np.linspace(0, w-1, new_w)
        y_new = np.linspace(0, h-1, new_h)
        
        # Indeksi centralnih piksela
        xi = np.floor(x_new).astype(int)
        yi = np.floor(y_new).astype(int)
        
        # Relativne pozicije unutar ćelije
        dx = x_new - xi
        dy = y_new - yi
        
        # Ograniči indekse
        xi = np.clip(xi, 0, w-1)
        yi = np.clip(yi, 0, h-1)
        
        result = np.zeros((new_h, new_w, c), dtype=np.float32)
        
        # Za svaki kanal
        for ch in range(c):
            channel = image[:, :, ch]
            output_channel = np.zeros((new_h, new_w), dtype=np.float32)
            
            # Za svaki red u izlaznoj slici
            for i in range(new_h):
                y = yi[i]
                dy_i = dy[i]
                
                # Uzmi 4 reda okoline
                rows = []
                for m in range(-1, 3):
                    row_idx = y + m
                    # Reflektuj ivice
                    if row_idx < 0:
                        row_idx = -row_idx
                    elif row_idx >= h:
                        row_idx = 2 * h - row_idx - 2
                    rows.append(channel[row_idx, :])
                
                # Interpolacija po kolonama za ovaj red
                for j in range(new_w):
                    x = xi[j]
                    dx_j = dx[j]
                    
                    # Uzmi 4 kolone
                    values = np.zeros(4)
                    for n in range(4):
                        col_idx = x + (n - 1)  # -1, 0, 1, 2
                        # Reflektuj ivice
                        if col_idx < 0:
                            col_idx = -col_idx
                        elif col_idx >= w:
                            col_idx = 2 * w - col_idx - 2
                        values[n] = rows[n][col_idx]
                    
                    # Interpolacija
                    output_channel[i, j] = cubic_interpolate(values, dx_j)
            
            result[:, :, ch] = output_channel
        
        return result
    else:
        # Za jednokanalne slike
        h, w = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        x_new = np.linspace(0, w-1, new_w)
        y_new = np.linspace(0, h-1, new_h)
        
        xi = np.floor(x_new).astype(int)
        yi = np.floor(y_new).astype(int)
        
        dx = x_new - xi
        dy = y_new - yi
        
        xi = np.clip(xi, 0, w-1)
        yi = np.clip(yi, 0, h-1)
        
        output = np.zeros((new_h, new_w), dtype=np.float32)
        
        # Koristi memoizaciju za redove
        row_cache = {}
        
        for i in range(new_h):
            y = yi[i]
            dy_i = dy[i]
            
            # Uzmi ili izračunaj 4 reda
            rows = []
            for m in range(-1, 3):
                row_idx = y + m
                # Reflektuj ivice
                if row_idx < 0:
                    row_idx = -row_idx
                elif row_idx >= h:
                    row_idx = 2 * h - row_idx - 2
                
                # Proveri cache
                if row_idx in row_cache:
                    rows.append(row_cache[row_idx])
                else:
                    rows.append(image[row_idx, :])
                    row_cache[row_idx] = image[row_idx, :]
            
            # Interpolacija po kolonama
            for j in range(new_w):
                x = xi[j]
                dx_j = dx[j]
                
                # Uzmi vrednosti iz 4 kolone
                values = np.array([
                    rows[0][max(0, min(x-1, w-1))],
                    rows[1][max(0, min(x, w-1))],
                    rows[2][max(0, min(x+1, w-1))],
                    rows[3][max(0, min(x+2, w-1))]
                ])
                
                output[i, j] = cubic_interpolate(values, dx_j)
        
        return output

def bicubic_reconstruct_optimized(image, missing_mask):
    """Optimizovana rekonstrukcija."""
    if len(image.shape) == 3:
        h, w, c = image.shape
        result = np.zeros_like(image)
        
        for ch in range(c):
            result[:, :, ch] = bicubic_reconstruct_channel_optimized(image[:, :, ch], missing_mask)
        
        return result
    else:
        return bicubic_reconstruct_channel_optimized(image, missing_mask)

def bicubic_reconstruct_channel_optimized(channel, missing_mask):
    """Rekonstrukcija sa vektorizovanom obradom."""
    h, w = channel.shape
    output = channel.copy()
    
    # Pronađi nedostajuće piksele
    missing_y, missing_x = np.where(missing_mask)
    
    if len(missing_y) == 0:
        return output
    
    # Grupiši po redovima za bolju lokalnost
    from collections import defaultdict
    rows_dict = defaultdict(list)
    
    for idx in range(len(missing_y)):
        rows_dict[missing_y[idx]].append(missing_x[idx])
    
    # Obradi red po red
    for y in sorted(rows_dict.keys()):
        x_list = rows_dict[y]
        
        for x in x_list:
            # Definiši 5x5 okolinu
            y_min, y_max = max(0, y-2), min(h, y+3)
            x_min, x_max = max(0, x-2), min(w, x+3)
            
            # Izvuci okolinu
            neighborhood = channel[y_min:y_max, x_min:x_max]
            neighborhood_mask = missing_mask[y_min:y_max, x_min:x_max]
            
            # Ukloni nedostajuće piksele
            valid_values = neighborhood[~neighborhood_mask]
            
            if len(valid_values) > 0:
                # Ponderisani prosek (bliži pikseli imaju veću težinu)
                # Kreiraj težine na osnovu udaljenosti
                center_y, center_x = y - y_min, x - x_min
                distances = np.sqrt(
                    (np.arange(neighborhood.shape[0]) - center_y)**2 + 
                    (np.arange(neighborhood.shape[1]) - center_x)**2
                )
                distances = distances[~neighborhood_mask]
                
                # Inverzna težinska funkcija
                weights = 1.0 / (distances + 1e-6)
                weights = weights / np.sum(weights)
                
                output[y, x] = np.dot(valid_values, weights)
    
    return output
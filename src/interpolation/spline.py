import numpy as np
from scipy.interpolate import RectBivariateSpline, griddata

def spline_interpolation(image, scale_factor=2, missing_mask=None):
    """
    OPTIMIZOVANA spline interpolacija.
    """
    if missing_mask is None:
        return spline_upscale(image, scale_factor)
    else:
        return spline_reconstruct(image, missing_mask)

def spline_upscale(image, scale_factor):
    """Spline interpolacija za upscaling."""
    if len(image.shape) == 3:
        h, w, c = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        result = np.zeros((new_h, new_w, c), dtype=np.float32)
        for ch in range(c):
            result[:, :, ch] = spline_upscale_channel(image[:, :, ch], scale_factor)
        return result
    else:
        return spline_upscale_channel(image, scale_factor)

def spline_upscale_channel(channel, scale_factor):
    """Spline interpolacija za jedan kanal."""
    h, w = channel.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Kreiraj spline interpolator
    x = np.arange(w)
    y = np.arange(h)
    spline = RectBivariateSpline(y, x, channel, kx=3, ky=3)
    
    # Kreiraj nove koordinate
    x_new = np.linspace(0, w-1, new_w)
    y_new = np.linspace(0, h-1, new_h)
    
    # Interpolacija
    return spline(y_new, x_new)

def spline_reconstruct(image, missing_mask):
    """Rekonstrukcija nedostajućih piksela spline-om."""
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for ch in range(image.shape[2]):
            result[:, :, ch] = spline_reconstruct_channel(image[:, :, ch], missing_mask)
        return result
    else:
        return spline_reconstruct_channel(image, missing_mask)

def spline_reconstruct_channel(channel, missing_mask):
    """Rekonstrukcija koristeći griddata (već optimizovano u scipy)."""
    h, w = channel.shape
    
    # Koordinate gde su pikseli poznati
    known_y, known_x = np.where(missing_mask == 0)
    known_values = channel[known_y, known_x]
    
    # Koordinate gde pikseli nedostaju
    missing_y, missing_x = np.where(missing_mask == 1)
    
    if len(missing_y) == 0:
        return channel.copy()
    
    # Interpolacija koristeći griddata sa linearnom metodom (brže od cubic)
    interp_values = griddata(
        (known_y, known_x), 
        known_values, 
        (missing_y, missing_x), 
        method='linear',  # možete promeniti u 'cubic' ako je potrebno
        fill_value=0.0
    )
    
    # Popuni nedostajuće vrednosti
    result = channel.copy()
    result[missing_y, missing_x] = interp_values
    
    return result
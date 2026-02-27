import numpy as np

def bilinear_interpolation(image, scale_factor=4, missing_mask=None):
    """Bilinearna interpolacija za uvecanje ili rekonstrukciju."""
    image = np.asarray(image, dtype=np.float32)

    if scale_factor <= 0:
        raise ValueError("scale_factor mora biti pozitivan broj.")

    if missing_mask is None:
        return bilinear_upscale_vectorized(image, scale_factor)
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
        result = np.empty((new_h, new_w, c), dtype=np.float32)
        
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

    # Za jednokanalne slike
    h, w = image.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    x = np.linspace(0, w - 1, new_w)
    y = np.linspace(0, h - 1, new_h)

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

    return (
        (1 - wy[:, None]) * (1 - wx) * q00
        + wy[:, None] * (1 - wx) * q10
        + (1 - wy[:, None]) * wx * q01
        + wy[:, None] * wx * q11
    )

def bilinear_reconstruct_vectorized(image, missing_mask):
    """Rekonstrukcija iterativnim lokalnim popunjavanjem."""
    missing_mask = np.asarray(missing_mask, dtype=bool)

    if image.shape[:2] != missing_mask.shape:
        raise ValueError("missing_mask mora imati iste dimenzije HxW kao slika.")

    if len(image.shape) == 3:
        _, _, c = image.shape
        result = np.empty_like(image, dtype=np.float32)
        
        for ch in range(c):
            result[:, :, ch] = bilinear_reconstruct_channel_vectorized(image[:, :, ch], missing_mask)

        return result
    return bilinear_reconstruct_channel_vectorized(image, missing_mask)

def bilinear_reconstruct_channel_vectorized(channel, missing_mask):
    """Rekonstrukcija distance-weighted prosekom suseda."""
    output = np.asarray(channel, dtype=np.float32).copy()
    remaining_mask = missing_mask.copy()

    if not np.any(remaining_mask):
        return output

    max_iterations = max(8, int(np.sqrt(output.shape[0] * output.shape[1]) // 2))
    for _ in range(max_iterations):
        missing_points = np.argwhere(remaining_mask)
        if missing_points.size == 0:
            break

        filled_count = 0
        for i, j in missing_points:
            value = _distance_weighted_fill(output, remaining_mask, i, j, max_radius=2)
            if value is not None:
                output[i, j] = value
                remaining_mask[i, j] = False
                filled_count += 1

        if filled_count == 0:
            break

    if np.any(remaining_mask):
        valid_values = output[~remaining_mask]
        fallback = np.mean(valid_values) if valid_values.size > 0 else 0.0
        output[remaining_mask] = fallback

    return output

def _distance_weighted_fill(image, missing_mask, i, j, max_radius=2):
    """Vraca interpoliranu vrednost ili None ako nema validnih suseda."""
    h, w = image.shape
    values = []
    weights = []

    for radius in range(1, max_radius + 1):
        i_min, i_max = max(0, i - radius), min(h, i + radius + 1)
        j_min, j_max = max(0, j - radius), min(w, j + radius + 1)

        block = image[i_min:i_max, j_min:j_max]
        block_mask = missing_mask[i_min:i_max, j_min:j_max]
        if np.all(block_mask):
            continue

        yy, xx = np.meshgrid(
            np.arange(i_min, i_max),
            np.arange(j_min, j_max),
            indexing="ij",
        )
        dist = np.sqrt((yy - i) ** 2 + (xx - j) ** 2)
        valid = (~block_mask) & (dist > 0)

        if np.any(valid):
            valid_dist = dist[valid]
            valid_vals = block[valid]
            w_local = 1.0 / valid_dist
            values.append(valid_vals)
            weights.append(w_local)

    if not values:
        return None

    all_values = np.concatenate(values)
    all_weights = np.concatenate(weights)
    all_weights /= np.sum(all_weights)
    return float(np.sum(all_values * all_weights))

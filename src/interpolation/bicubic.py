import numpy as np

def cubic_weights_catmull_rom(t):
    """Catmull-Rom tezine za offsete [-1, 0, 1, 2]."""
    t2 = t * t
    t3 = t2 * t
    w0 = -0.5 * t + t2 - 0.5 * t3
    w1 = 1.0 - 2.5 * t2 + 1.5 * t3
    w2 = 0.5 * t + 2.0 * t2 - 1.5 * t3
    w3 = -0.5 * t2 + 0.5 * t3
    return np.stack((w0, w1, w2, w3), axis=-1)

def reflect_indices(indices, size):
    """Refleksija indeksa preko granica slike."""
    if size <= 1:
        return np.zeros_like(indices, dtype=np.int32)
    period = 2 * size - 2
    indices_mod = np.mod(indices, period)
    reflected = np.where(indices_mod < size, indices_mod, period - indices_mod)
    return reflected.astype(np.int32)

def bicubic_interpolation(image, scale_factor=4, missing_mask=None):
    """Bikubna interpolacija za uvecanje ili rekonstrukciju."""
    image = np.asarray(image, dtype=np.float32)

    if scale_factor <= 0:
        raise ValueError("scale_factor mora biti pozitivan broj.")

    if missing_mask is None:
        return bicubic_upscale_optimized(image, scale_factor)
    return bicubic_reconstruct_optimized(image, missing_mask)

def bicubic_upscale_optimized(image, scale_factor):
    """Vektorizovana separabilna bikubna interpolacija."""
    if len(image.shape) == 3:
        h, w, c = image.shape
        result = np.empty((int(h * scale_factor), int(w * scale_factor), c), dtype=np.float32)

        for ch in range(c):
            result[:, :, ch] = _bicubic_upscale_channel(image[:, :, ch], scale_factor)
        return result

    return _bicubic_upscale_channel(image, scale_factor)

def _bicubic_upscale_channel(channel, scale_factor):
    """Bikubno uvecanje jednog kanala."""
    h, w = channel.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    x_new = np.linspace(0, w - 1, new_w, dtype=np.float32)
    y_new = np.linspace(0, h - 1, new_h, dtype=np.float32)

    xi = np.floor(x_new).astype(np.int32)
    yi = np.floor(y_new).astype(np.int32)
    tx = x_new - xi
    ty = y_new - yi

    x_idx = reflect_indices(xi[:, None] + np.array([-1, 0, 1, 2], dtype=np.int32), w)
    y_idx = reflect_indices(yi[:, None] + np.array([-1, 0, 1, 2], dtype=np.int32), h)

    wx = cubic_weights_catmull_rom(tx)
    wy = cubic_weights_catmull_rom(ty)

    # 1) Interpolacija po x-osi za sve redove
    temp = np.zeros((h, new_w), dtype=np.float32)
    for k in range(4):
        temp += channel[:, x_idx[:, k]] * wx[:, k]

    # 2) Interpolacija po y-osi
    output = np.zeros((new_h, new_w), dtype=np.float32)
    for k in range(4):
        output += wy[:, k][:, None] * temp[y_idx[:, k], :]

    return output

def bicubic_reconstruct_optimized(image, missing_mask):
    """Rekonstrukcija lokalnim ponderisanim kubnim popunjavanjem."""
    missing_mask = np.asarray(missing_mask, dtype=bool)

    if image.shape[:2] != missing_mask.shape:
        raise ValueError("missing_mask mora imati iste dimenzije HxW kao slika.")

    if len(image.shape) == 3:
        _, _, c = image.shape
        result = np.empty_like(image, dtype=np.float32)
        for ch in range(c):
            result[:, :, ch] = bicubic_reconstruct_channel_optimized(image[:, :, ch], missing_mask)

        return result

    return bicubic_reconstruct_channel_optimized(image, missing_mask)

def bicubic_reconstruct_channel_optimized(channel, missing_mask):
    """Iterativna rekonstrukcija sa distancnim ponderisanjem."""
    output = np.asarray(channel, dtype=np.float32).copy()
    remaining_mask = missing_mask.copy()

    if not np.any(remaining_mask):
        return output

    max_iterations = max(10, int(np.sqrt(output.shape[0] * output.shape[1]) // 2))
    for _ in range(max_iterations):
        missing_points = np.argwhere(remaining_mask)
        if missing_points.size == 0:
            break

        filled_count = 0
        for i, j in missing_points:
            value = _distance_weighted_fill(output, remaining_mask, i, j, max_radius=3, power=2.0)
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

def _distance_weighted_fill(image, missing_mask, i, j, max_radius=3, power=2.0):
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
            w_local = 1.0 / (valid_dist ** power)
            values.append(valid_vals)
            weights.append(w_local)

    if not values:
        return None

    all_values = np.concatenate(values)
    all_weights = np.concatenate(weights)
    all_weights /= np.sum(all_weights)
    return float(np.sum(all_values * all_weights))

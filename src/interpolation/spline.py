import numpy as np

def spline_interpolation(image, scale_factor=4, missing_mask=None):
    """Kubna spline interpolacija za uvecanje ili rekonstrukciju."""
    image = np.asarray(image, dtype=np.float32)

    if scale_factor <= 0:
        raise ValueError("scale_factor mora biti pozitivan broj.")

    if missing_mask is None:
        return spline_upscale_fast(image, scale_factor)
    return spline_reconstruct_fast(image, missing_mask)

def spline_upscale_fast(image, scale_factor):
    """Separabilno uvecanje natural cubic spline metodom."""
    if len(image.shape) == 3:
        h, w, c = image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        result = np.empty((new_h, new_w, c), dtype=np.float32)

        x_old = np.arange(w, dtype=np.float32)
        x_new = np.linspace(0, w - 1, new_w, dtype=np.float32)
        y_old = np.arange(h, dtype=np.float32)
        y_new = np.linspace(0, h - 1, new_h, dtype=np.float32)

        for ch in range(c):
            channel = image[:, :, ch]
            temp = np.empty((h, new_w), dtype=np.float32)

            for row in range(h):
                temp[row, :] = natural_cubic_spline_eval(x_old, channel[row, :], x_new)

            for col in range(new_w):
                result[:, col, ch] = natural_cubic_spline_eval(y_old, temp[:, col], y_new)

        return result

    h, w = image.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    temp = np.empty((h, new_w), dtype=np.float32)
    x_old = np.arange(w, dtype=np.float32)
    x_new = np.linspace(0, w - 1, new_w, dtype=np.float32)

    for row in range(h):
        temp[row, :] = natural_cubic_spline_eval(x_old, image[row, :], x_new)

    result = np.empty((new_h, new_w), dtype=np.float32)
    y_old = np.arange(h, dtype=np.float32)
    y_new = np.linspace(0, h - 1, new_h, dtype=np.float32)

    for col in range(new_w):
        result[:, col] = natural_cubic_spline_eval(y_old, temp[:, col], y_new)

    return result

def spline_reconstruct_fast(image, missing_mask):
    """Rekonstrukcija koristeci 1D kubne splajnove po redovima i kolonama."""
    missing_mask = np.asarray(missing_mask, dtype=bool)

    if image.shape[:2] != missing_mask.shape:
        raise ValueError("missing_mask mora imati iste dimenzije HxW kao slika.")

    if len(image.shape) == 3:
        _, _, c = image.shape
        result = np.empty_like(image, dtype=np.float32)

        for ch in range(c):
            result[:, :, ch] = spline_reconstruct_channel_fast(image[:, :, ch], missing_mask)

        return result

    return spline_reconstruct_channel_fast(image, missing_mask)

def spline_reconstruct_channel_fast(channel, missing_mask):
    """Iterativna rekonstrukcija kubnim splajnovima duz redova i kolona."""
    h, w = channel.shape
    output = np.asarray(channel, dtype=np.float32).copy()
    remaining_mask = missing_mask.copy()

    if not np.any(remaining_mask):
        return output

    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)

    max_iterations = 6
    for _ in range(max_iterations):
        filled_count = 0

        # Pass 1: redovi
        for i in range(h):
            row_missing = remaining_mask[i, :]
            if not np.any(row_missing):
                continue

            known = ~row_missing
            known_count = int(np.sum(known))
            if known_count < 2:
                continue

            x_known = x_coords[known]
            y_known = output[i, known]
            x_query = x_coords[row_missing]
            output[i, row_missing] = _safe_spline_eval(x_known, y_known, x_query)
            remaining_mask[i, row_missing] = False
            filled_count += int(np.sum(row_missing))

        # Pass 2: kolone za preostale
        for j in range(w):
            col_missing = remaining_mask[:, j]
            if not np.any(col_missing):
                continue

            known = ~col_missing
            known_count = int(np.sum(known))
            if known_count < 2:
                continue

            y_known = y_coords[known]
            v_known = output[known, j]
            y_query = y_coords[col_missing]
            output[col_missing, j] = _safe_spline_eval(y_known, v_known, y_query)
            remaining_mask[col_missing, j] = False
            filled_count += int(np.sum(col_missing))

        if filled_count == 0:
            break

    # Fallback za retke slucajeve gde nema dovoljno tacaka
    if np.any(remaining_mask):
        valid_values = output[~remaining_mask]
        fallback = np.mean(valid_values) if valid_values.size > 0 else 0.0
        output[remaining_mask] = fallback

    return output

def _safe_spline_eval(x_known, y_known, x_query):
    """Spline evaluacija sa fallback-om na linearnu interpolaciju."""
    if x_known.size < 2:
        return np.full_like(x_query, y_known[0] if y_known.size > 0 else 0.0, dtype=np.float32)

    # Za vrlo mali broj tacaka linearna je stabilnija
    if x_known.size < 4:
        return np.interp(x_query, x_known, y_known).astype(np.float32)

    return natural_cubic_spline_eval(x_known, y_known, x_query)

def natural_cubic_spline_eval(x, y, xq):
    """Natural cubic spline evaluacija za opste (strogo rastuce) cvorove."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    xq = np.asarray(xq, dtype=np.float32)

    n = x.size
    if n == 0:
        return np.zeros_like(xq, dtype=np.float32)
    if n == 1:
        return np.full_like(xq, y[0], dtype=np.float32)

    # Osiguraj rastuce i jedinstvene cvorove
    keep = np.r_[True, np.diff(x) > 0]
    x = x[keep]
    y = y[keep]
    n = x.size

    if n < 2:
        return np.full_like(xq, y[0], dtype=np.float32)
    if n < 4:
        return np.interp(xq, x, y).astype(np.float32)

    m = _natural_second_derivatives(x, y)

    # Izbor segmenta za svaku query tacku
    idx = np.searchsorted(x, xq, side="right") - 1
    idx = np.clip(idx, 0, n - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    m0 = m[idx]
    m1 = m[idx + 1]

    h = x1 - x0
    # zastita od nule (iako je vec filtrirano)
    h = np.where(h == 0, 1.0, h)

    a = (x1 - xq) / h
    b = (xq - x0) / h

    sq = (
        m0 * ((a ** 3 - a) * (h ** 2) / 6.0)
        + m1 * ((b ** 3 - b) * (h ** 2) / 6.0)
        + y0 * a
        + y1 * b
    )
    return sq.astype(np.float32)

def _natural_second_derivatives(x, y):
    """Racuna druge izvode m_i za natural cubic spline."""
    n = x.size
    m = np.zeros(n, dtype=np.float32)
    if n < 3:
        return m

    h = np.diff(x)
    lower = h[:-1].copy()
    diag = 2.0 * (h[:-1] + h[1:])
    upper = h[1:].copy()
    rhs = 6.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])

    sol = _solve_tridiagonal(lower, diag, upper, rhs)
    m[1:-1] = sol
    return m

def _solve_tridiagonal(a, b, c, d):
    """Thomas algoritam za tridiagonalni sistem."""
    n = b.size
    if n == 0:
        return np.array([], dtype=np.float32)

    ac = a.astype(np.float32).copy()
    bc = b.astype(np.float32).copy()
    cc = c.astype(np.float32).copy()
    dc = d.astype(np.float32).copy()

    for i in range(1, n):
        w = ac[i - 1] / bc[i - 1]
        bc[i] = bc[i] - w * cc[i - 1]
        dc[i] = dc[i] - w * dc[i - 1]

    x = np.empty(n, dtype=np.float32)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]

    return x

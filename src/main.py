import numpy as np
import torch
from pathlib import Path

from src.utils.image_io import load_image, save_image
from src.utils.image_io import rgb_to_ycbcr, ycbcr_to_rgb
from src.interpolation.bicubic import bicubic_interpolation
from src.interpolation.bilinear import bilinear_upscale_vectorized
from src.interpolation.spline import spline_upscale_fast
from src.neural_network.srcnn_model import SRCNN


SCALE_FACTOR = 3
BEST_MODEL_PATH = Path("models/srcnn_y_best.pth")
LAST_MODEL_PATH = Path("models/srcnn_y.pth")
INPUT_PATH = Path("user/input/input_image.png")
OUTPUT_DIR = Path("user/output")


def _upscale_channel(channel, method, scale_factor):
    if method == "bicubic":
        return bicubic_interpolation(channel, scale_factor=scale_factor)
    if method == "bilinear":
        return bilinear_upscale_vectorized(channel, scale_factor=scale_factor)
    if method == "spline":
        return spline_upscale_fast(channel, scale_factor=scale_factor)
    raise ValueError(f"Nepoznata interpolaciona metoda: {method}")


def _load_srcnn(model_path, device):
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def _resolve_model_path():
    if BEST_MODEL_PATH.exists():
        return BEST_MODEL_PATH
    if LAST_MODEL_PATH.exists():
        return LAST_MODEL_PATH
    raise FileNotFoundError("Model nije pronađen. Pokreni trening pre SRCNN inferencije.")


def _enhance_y_with_srcnn(y_channel, model, device):
    y_tensor = torch.from_numpy(y_channel / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        y_sr = model(y_tensor).squeeze().cpu().numpy() * 255.0
    return np.clip(y_sr, 0.0, 255.0)


def main():
    img = load_image(INPUT_PATH)
    print("Original shape:", img.shape)

    print("\n========== Upscaling ==========")
    print("Opcije interpolacije:")
    print("1 - Bicubic")
    print("2 - Bilinear")
    print("3 - Spline")
    print("4 - Bicubic + SRCNN (Y-channel)")
    print("5 - Spline + SRCNN (Y-channel)")
    print("6 - Bilinear + SRCNN (Y-channel)")
    print("===============================")

    opcija = int(input("Unesite opciju (1-6): "))

    # RGB → YCbCr
    y, cb, cr = rgb_to_ycbcr(img)

    options = {
        1: ("bicubic", False, "output_bicubic.png"),
        2: ("bilinear", False, "output_bilinear.png"),
        3: ("spline", False, "output_spline.png"),
        4: ("bicubic", True, "output_bicubic_srcnn.png"),
        5: ("spline", True, "output_spline_srcnn.png"),
        6: ("bilinear", True, "output_bilinear_srcnn.png"),
    }

    selected = options.get(opcija)
    if selected is None:
        print("Nepoznata opcija!")
        return

    method, use_srcnn, output_name = selected
    y_up = _upscale_channel(y, method, SCALE_FACTOR)
    cb_up = _upscale_channel(cb, method, SCALE_FACTOR)
    cr_up = _upscale_channel(cr, method, SCALE_FACTOR)

    if use_srcnn:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = _resolve_model_path()
        model = _load_srcnn(model_path, device)
        y_up = _enhance_y_with_srcnn(y_up, model, device)

    # Spoji kanale i sacuvaj
    sr_rgb = ycbcr_to_rgb(y_up, cb_up, cr_up).astype(np.uint8)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / output_name
    save_image(sr_rgb, out_path)
    print(f"Output saved: {out_path}")


if __name__ == "__main__":
    main()

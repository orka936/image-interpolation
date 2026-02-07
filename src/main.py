import numpy as np
import torch
from PIL import Image

from src.utils.image_io import load_image, save_image
from src.interpolation.bicubic import bicubic_interpolation
from src.interpolation.bilinear import bilinear_upscale_vectorized
from src.interpolation.spline import spline_upscale_fast
from src.neural_network.srcnn_model import SRCNN


def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    y  = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    cb = -0.1687 * img[:,:,0] - 0.3313 * img[:,:,1] + 0.5 * img[:,:,2] + 128
    cr = 0.5 * img[:,:,0] - 0.4187 * img[:,:,1] - 0.0813 * img[:,:,2] + 128
    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    img = np.stack([r, g, b], axis=2)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    img = load_image("user/input/input_image.png")
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

    SCALE = 4  # primer, zavisi od tvog treniranja

    if opcija == 1:
        y_up = bicubic_interpolation(y, scale_factor=SCALE)
        cb_up = bicubic_interpolation(cb, scale_factor=SCALE)
        cr_up = bicubic_interpolation(cr, scale_factor=SCALE)
        out_name = "user/output/output_bicubic.png"

    elif opcija == 2:
        y_up = bilinear_upscale_vectorized(y, scale_factor=SCALE)
        cb_up = bilinear_upscale_vectorized(cb, scale_factor=SCALE)
        cr_up = bilinear_upscale_vectorized(cr, scale_factor=SCALE)
        out_name = "user/output/output_bilinear.png"

    elif opcija == 3:
        y_up = spline_upscale_fast(y, scale_factor=SCALE)
        cb_up = spline_upscale_fast(cb, scale_factor=SCALE)
        cr_up = spline_upscale_fast(cr, scale_factor=SCALE)
        out_name = "user/output/output_spline.png"

    # SRCNN opcije
    elif opcija == 4:
        y_up  = bicubic_interpolation(y, scale_factor=SCALE)
        cb_up = bicubic_interpolation(cb, scale_factor=SCALE)
        cr_up = bicubic_interpolation(cr, scale_factor=SCALE)
        y_tensor = torch.from_numpy(y_up / 255.0).float().unsqueeze(0).unsqueeze(0)
        model = SRCNN()
        model.load_state_dict(torch.load("models/srcnn_y.pth", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            y_up = model(y_tensor).squeeze().numpy() * 255.0
        out_name = "user/output/output_bicubic_srcnn.png"

    elif opcija == 5:
        y_up  = spline_upscale_fast(y, scale_factor=SCALE)
        cb_up = spline_upscale_fast(cb, scale_factor=SCALE)
        cr_up = spline_upscale_fast(cr, scale_factor=SCALE)
        y_tensor = torch.from_numpy(y_up / 255.0).float().unsqueeze(0).unsqueeze(0)
        model = SRCNN()
        model.load_state_dict(torch.load("models/srcnn_y.pth", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            y_up = model(y_tensor).squeeze().numpy() * 255.0
        out_name = "user/output/output_spline_srcnn.png"

    elif opcija == 6:
        y_up  = bilinear_upscale_vectorized(y, scale_factor=SCALE)
        cb_up = bilinear_upscale_vectorized(cb, scale_factor=SCALE)
        cr_up = bilinear_upscale_vectorized(cr, scale_factor=SCALE)
        y_tensor = torch.from_numpy(y_up / 255.0).float().unsqueeze(0).unsqueeze(0)
        model = SRCNN()
        model.load_state_dict(torch.load("models/srcnn_y.pth", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            y_up = model(y_tensor).squeeze().numpy() * 255.0
        out_name = "user/output/output_bilinear_srcnn.png"

    else:
        print("Nepoznata opcija!")
        return

    # Spoji kanale i sacuvaj
    sr_rgb = ycbcr_to_rgb(y_up, cb_up, cr_up)
    save_image(sr_rgb, out_name)
    print(f"Output saved: {out_name}")


if __name__ == "__main__":
    main()

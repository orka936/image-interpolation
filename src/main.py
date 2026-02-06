import numpy as np
from utils.image_io import load_image, save_image
from interpolation.bilinear import bilinear_interpolation
from interpolation.bicubic import bicubic_interpolation
from interpolation.spline import spline_interpolation

def main():
    # Učitaj sliku
    img = load_image("user/input/input_image.png")  # (H, W, 3) float32
    print(f"Original slika shape: {img.shape}")

    # 1. Povećanje rezolucije (UPSACLING)
    print("Upscaling...")
    img_bilinear = bilinear_interpolation(img)
    img_bicubic = bicubic_interpolation(img)
    img_spline = spline_interpolation(img)

    save_image(img_bilinear, "user/output/output_bilinear.png")
    save_image(img_bicubic, "user/output/output_bicubic.png")
    save_image(img_spline, "user/output/output_spline.png")
    print("Upscaled slike sačuvane.")

    # 2. Rekonstrukcija nedostajućih piksela (isti razmer)
    print("Rekonstrukcija nedostajućih piksela...")
    h, w, _ = img.shape
    missing_mask = np.random.rand(h, w) > 0.9  # 10% piksela nedostaje
    img_missing = img.copy()
    img_missing[missing_mask == 1] = 0  # postavi nedostajuće piksele na 0

    # BITNO: ne prosleđujemo scale_factor za rekonstrukciju
    recon_bilinear = bilinear_interpolation(img_missing, missing_mask=missing_mask)
    recon_bicubic = bicubic_interpolation(img_missing, missing_mask=missing_mask)
    recon_spline = spline_interpolation(img_missing, missing_mask=missing_mask)

    save_image(recon_bilinear, "user/output_recon/recon_bilinear.png")
    save_image(recon_bicubic, "user/output_recon/recon_bicubic.png")
    save_image(recon_spline, "user/output_recon/recon_spline.png")
    print("Rekonstruisane slike sačuvane.")

if __name__ == "__main__":
    main()
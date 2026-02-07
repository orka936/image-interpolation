import numpy as np
from utils.image_io import load_image, save_image
from interpolation.bilinear import bilinear_interpolation
from interpolation.bicubic import bicubic_interpolation
from interpolation.spline import spline_interpolation

def main():

    # Učitaj sliku - OVO ĆE RADITI
    img = load_image("user/input/input_image.png")
    print(f"\nOriginal slika shape: {img.shape}")

    print("\n========== Upscaling ==========")
    print("1 Bilinearna interpolacija")
    print("2 Bikubna interpolacija")
    print("3 Spline interpolacija")
    print("===============================")
    opcija=int(input("Izaberite opciju za upscaling: "))

    # 1. Upscaling - OVO ĆE RADITI
    print("\nUpscaling...")
    
    if opcija==1:
        img_bilinear = bilinear_interpolation(img)
        save_image(img_bilinear, "user/output/output_bilinear.png")

    if opcija==2:
        img_bicubic = bicubic_interpolation(img)
        save_image(img_bicubic, "user/output/output_bicubic.png")

    if opcija==3:
        img_spline = spline_interpolation(img)
        save_image(img_spline, "user/output/output_spline.png")

    print("Upscaled slika je sačuvana.")

    h, w, _ = img.shape
    missing_mask = np.random.rand(h, w) > 0.9
    img_missing = img.copy()
    img_missing[missing_mask == 1] = 0

    print("\n===== Rekonstrukcija piksela =====")
    print("1 Bilinearna rekonstrukcija")
    print("2 Bikubna rekonstrukcija")
    print("3 Spline rekonstrukcija")
    print("===================================")
    opcija=int(input("\nOpcija za rekonstrukciju piksela: "))
    print("\nRekonstrukcija nedostajućih piksela...")

    if opcija==1:
        recon_bilinear = bilinear_interpolation(img_missing, missing_mask=missing_mask)
        save_image(recon_bilinear, "user/output_recon/recon_bilinear.png")

    if opcija==2:
        recon_bicubic = bicubic_interpolation(img_missing, missing_mask=missing_mask)
        save_image(recon_bicubic, "user/output_recon/recon_bicubic.png")

    if opcija==3:
        recon_spline = spline_interpolation(img_missing, missing_mask=missing_mask)
        save_image(recon_spline, "user/output_recon/recon_spline.png")

    print("Rekonstruisana slika je sačuvana.\n")

if __name__ == "__main__":
    main()
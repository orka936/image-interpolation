import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path

from src.utils.image_io import load_image, save_image
from src.utils.degradation import downsample_image


# Podesavanja
INPUT_DIR = Path("data/input/original")
OUTPUT_DIR = Path("data/input/degraded")
SCALE_FACTOR = 3   # 3x downscale


def generate_degraded_images():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for img_name in os.listdir(INPUT_DIR):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        input_path = INPUT_DIR / img_name
        output_path = OUTPUT_DIR / img_name

        print(f"Degradiram: {img_name}")

        image = load_image(input_path)
        degraded = downsample_image(image, SCALE_FACTOR)
        save_image(degraded, output_path)


if __name__ == "__main__":
    generate_degraded_images()

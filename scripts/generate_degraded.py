import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path

from src.utils.image_io import load_image, save_image
from src.utils.degradation import downsample_image


# Podesavanja
INPUT_DIR = Path("data/input/original")
OUTPUT_DIR = Path("data/input/degraded")
SCALE_FACTOR = 3


def generate_degraded_images(input_dir: Path, output_dir: Path, scale_factor: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        input_path = input_dir / img_name
        output_path = output_dir / img_name

        print(f"Degradiram: {img_name}")

        image = load_image(input_path)
        degraded = downsample_image(image, scale_factor)
        save_image(degraded, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generisanje degradiranih slika downsample metodom.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="Direktorijum originalnih slika")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Direktorijum degradiranih slika")
    parser.add_argument("--scale", type=int, default=SCALE_FACTOR, help="Scale faktor za downsample")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_degraded_images(args.input_dir, args.output_dir, args.scale)

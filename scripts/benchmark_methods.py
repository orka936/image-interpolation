import sys
import os
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.image_io import load_image, rgb_to_ycbcr, ycbcr_to_rgb
from src.interpolation.bicubic import bicubic_interpolation
from src.interpolation.bilinear import bilinear_upscale_vectorized
from src.interpolation.spline import spline_upscale_fast
from src.neural_network.srcnn_model import SRCNN
from src.evaluation.metrics import psnr


def _upscale_channel(channel, method, scale_factor):
    if method == "bicubic":
        return bicubic_interpolation(channel, scale_factor=scale_factor)
    if method == "bilinear":
        return bilinear_upscale_vectorized(channel, scale_factor=scale_factor)
    if method == "spline":
        return spline_upscale_fast(channel, scale_factor=scale_factor)
    raise ValueError(f"Nepoznata metoda: {method}")


def _apply_srcnn_y(y_up, model, device):
    y_tensor = torch.from_numpy(y_up / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        y_sr = model(y_tensor).squeeze().cpu().numpy() * 255.0
    return np.clip(y_sr, 0.0, 255.0)


def _resolve_scale(lr_shape, hr_shape, explicit_scale=None):
    lr_h, lr_w = lr_shape[:2]
    hr_h, hr_w = hr_shape[:2]

    if explicit_scale is not None:
        return float(explicit_scale)

    if lr_h == 0 or lr_w == 0:
        raise ValueError("LR slika ima nevalidne dimenzije.")

    scale_h = hr_h / lr_h
    scale_w = hr_w / lr_w
    if abs(scale_h - scale_w) > 1e-6:
        raise ValueError(f"Neusklađen scale između osa: h={scale_h:.4f}, w={scale_w:.4f}")
    return float(scale_h)


def _pair_images(lr_dir, hr_dir):
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp"}
    lr_files = {
        p.stem: p
        for p in Path(lr_dir).glob("*")
        if p.is_file() and p.suffix.lower() in valid_ext
    }
    hr_files = {
        p.stem: p
        for p in Path(hr_dir).glob("*")
        if p.is_file() and p.suffix.lower() in valid_ext
    }

    keys = sorted(set(lr_files.keys()) & set(hr_files.keys()))
    return [(lr_files[k], hr_files[k]) for k in keys]


def _reconstruct_rgb(lr_rgb, method, scale_factor, srcnn_model=None, device="cpu"):
    y, cb, cr = rgb_to_ycbcr(lr_rgb)

    y_up = _upscale_channel(y, method, scale_factor)
    cb_up = _upscale_channel(cb, method, scale_factor)
    cr_up = _upscale_channel(cr, method, scale_factor)

    if srcnn_model is not None:
        y_up = _apply_srcnn_y(y_up, srcnn_model, device)

    return ycbcr_to_rgb(y_up, cb_up, cr_up).astype(np.uint8)


def run_benchmark(lr_dir, hr_dir, model_path=None, scale=None, csv_path=None):
    pairs = _pair_images(lr_dir, hr_dir)
    if not pairs:
        raise ValueError("Nema zajedničkih LR/HR fajlova za benchmark.")

    methods = [
        ("bicubic", False),
        ("bilinear", False),
        ("spline", False),
        ("bicubic", True),
        ("bilinear", True),
        ("spline", True),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    srcnn_model = None
    if any(use_srcnn for _, use_srcnn in methods):
        if model_path is None:
            best = Path("models/srcnn_y_best.pth")
            fallback = Path("models/srcnn_y.pth")
            model_path = best if best.exists() else fallback

        srcnn_model = SRCNN().to(device)
        srcnn_model.load_state_dict(torch.load(model_path, map_location=device))
        srcnn_model.eval()

    rows = []
    agg = {}

    for lr_path, hr_path in pairs:
        lr = load_image(lr_path)
        hr = load_image(hr_path)

        local_scale = _resolve_scale(lr.shape, hr.shape, explicit_scale=scale)

        for method, use_srcnn in methods:
            tag = f"{method}{'_srcnn' if use_srcnn else ''}"
            pred = _reconstruct_rgb(
                lr,
                method=method,
                scale_factor=local_scale,
                srcnn_model=srcnn_model if use_srcnn else None,
                device=device,
            )

            if pred.shape != hr.shape:
                h = min(pred.shape[0], hr.shape[0])
                w = min(pred.shape[1], hr.shape[1])
                pred_eval = pred[:h, :w]
                hr_eval = hr[:h, :w]
            else:
                pred_eval = pred
                hr_eval = hr

            value_psnr = psnr(pred_eval, hr_eval, data_range=255.0)
            value_mse = float(np.mean((pred_eval.astype(np.float64) - hr_eval.astype(np.float64)) ** 2))

            rows.append({
                "image": lr_path.stem,
                "method": tag,
                "psnr": value_psnr,
                "mse": value_mse,
            })

            agg.setdefault(tag, {"psnr": [], "mse": []})
            agg[tag]["psnr"].append(value_psnr)
            agg[tag]["mse"].append(value_mse)

    print("\n===== Prosečni rezultati =====")
    for tag, values in agg.items():
        avg_psnr = float(np.mean(values["psnr"])) if values["psnr"] else 0.0
        avg_mse = float(np.mean(values["mse"])) if values["mse"] else 0.0
        print(f"{tag:16s} | PSNR: {avg_psnr:6.2f} dB | MSE: {avg_mse:8.2f}")

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "method", "psnr", "mse"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSačuvani detaljni rezultati: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark interpolacija i SRCNN varijanti.")
    parser.add_argument("--lr-dir", type=Path, default=Path("data/input/degraded"), help="Direktorijum sa LR slikama")
    parser.add_argument("--hr-dir", type=Path, default=Path("data/input/original"), help="Direktorijum sa HR slikama")
    parser.add_argument("--model", type=Path, default=None, help="Putanja do SRCNN modela")
    parser.add_argument("--scale", type=float, default=3.0, help="Scale faktor")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("user/output_recon/benchmark_results.csv"),
        help="Putanja za CSV izveštaj",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        lr_dir=args.lr_dir,
        hr_dir=args.hr_dir,
        model_path=args.model,
        scale=args.scale,
        csv_path=args.csv,
    )
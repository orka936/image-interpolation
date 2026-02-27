import numpy as np
import torch
from scipy.ndimage import uniform_filter


def _to_numpy_float32(image):
	if isinstance(image, torch.Tensor):
		image = image.detach().cpu().numpy()
	arr = np.asarray(image, dtype=np.float32)
	return arr


def _prepare_pair(pred, target, data_range=255.0):
	pred_arr = _to_numpy_float32(pred)
	target_arr = _to_numpy_float32(target)

	if pred_arr.shape != target_arr.shape:
		raise ValueError(f"Shape mismatch: pred={pred_arr.shape}, target={target_arr.shape}")

	pred_arr = np.clip(pred_arr, 0.0, data_range)
	target_arr = np.clip(target_arr, 0.0, data_range)
	return pred_arr, target_arr


def psnr(pred, target, data_range=255.0, eps=1e-12):
	pred_arr, target_arr = _prepare_pair(pred, target, data_range=data_range)
	mse = np.mean((pred_arr - target_arr) ** 2)
	mse = max(mse, eps)
	return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def ssim(pred, target, data_range=255.0, k1=0.01, k2=0.03, win_size=11):
	"""Strukturna sličnost sa lokalnim prozorom (kao u originalu: Wang et al. 2004)."""
	pred_arr, target_arr = _prepare_pair(pred, target, data_range=data_range)

	if pred_arr.ndim == 3 and pred_arr.shape[2] in (1, 3):
		channel_scores = [
			_ssim_single_channel(pred_arr[:, :, ch], target_arr[:, :, ch],
			                     data_range, k1, k2, win_size)
			for ch in range(pred_arr.shape[2])
		]
		return float(np.mean(channel_scores))

	if pred_arr.ndim != 2:
		raise ValueError("SSIM ocekuje 2D sliku ili 3D sliku sa 1/3 kanala.")

	return _ssim_single_channel(pred_arr, target_arr, data_range, k1, k2, win_size)


def _ssim_single_channel(pred, target, data_range, k1, k2, win_size):
	"""SSIM sa lokalnim uniform filterom (aproksimacija Gausovog prozora)."""
	c1 = (k1 * data_range) ** 2
	c2 = (k2 * data_range) ** 2

	mu_x = uniform_filter(pred, size=win_size)
	mu_y = uniform_filter(target, size=win_size)

	mu_x_sq = mu_x ** 2
	mu_y_sq = mu_y ** 2
	mu_xy = mu_x * mu_y

	sigma_x_sq = uniform_filter(pred ** 2, size=win_size) - mu_x_sq
	sigma_y_sq = uniform_filter(target ** 2, size=win_size) - mu_y_sq
	sigma_xy = uniform_filter(pred * target, size=win_size) - mu_xy

	# Clamp negativne variance (numericka greska)
	sigma_x_sq = np.maximum(sigma_x_sq, 0.0)
	sigma_y_sq = np.maximum(sigma_y_sq, 0.0)

	numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
	denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

	ssim_map = numerator / denominator
	# Crop border (half window on each side)
	pad = win_size // 2
	return float(np.mean(ssim_map[pad:-pad, pad:-pad]))

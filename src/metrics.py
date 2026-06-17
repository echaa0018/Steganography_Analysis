from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def mse(cover: np.ndarray, stego: np.ndarray) -> float:
    a = np.asarray(cover, dtype=np.float64)
    b = np.asarray(stego, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr(cover: np.ndarray, stego: np.ndarray, max_value: float = 255.0) -> float:
    error = mse(cover, stego)
    if error == 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_value ** 2) / error))


def ssim(cover: np.ndarray, stego: np.ndarray) -> float:
    a = np.asarray(cover)
    b = np.asarray(stego)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    channel_axis = -1 if a.ndim == 3 else None
    value = structural_similarity(
        a, b, channel_axis=channel_axis, data_range=255
    )
    return float(value)


def payload_bits_per_pixel(n_bits: int, image: np.ndarray) -> float:
    img = np.asarray(image)
    n_pixels = img.shape[0] * img.shape[1]
    return float(n_bits) / float(n_pixels)


def payload_capacity(image: np.ndarray, n_bits: int) -> dict[str, float]:
    return {
        "payload_bits": float(n_bits),
        "payload_bytes": float(n_bits) / 8.0,
        "bpp": payload_bits_per_pixel(n_bits, image),
    }


def compute_imperceptibility(cover: np.ndarray, stego: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(cover, stego),
        "psnr": psnr(cover, stego),
        "ssim": ssim(cover, stego),
    }

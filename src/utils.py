from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]


def load_image(path: PathLike, mode: str | None = None) -> np.ndarray:
    with Image.open(path) as img:
        if mode is not None:
            img = img.convert(mode)
        return np.asarray(img, dtype=np.uint8)


def save_image(array: np.ndarray, path: PathLike) -> None:
    arr = np.asarray(array)
    if not np.issubdtype(arr.dtype, np.uint8):
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(Path(path))


def to_grayscale(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float64)
        luma = rgb @ np.array([0.299, 0.587, 0.114])
        return np.clip(np.rint(luma), 0, 255).astype(np.uint8)
    raise ValueError(f"Expected a 2D or 3D image array, got shape {arr.shape}")


def to_rgb(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1).astype(np.uint8)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            return arr[..., :3].astype(np.uint8)
        if arr.shape[2] == 1:
            return np.repeat(arr, 3, axis=2).astype(np.uint8)
    raise ValueError(f"Cannot convert array of shape {arr.shape} to RGB")


def make_synthetic_image(
    height: int = 256,
    width: int = 256,
    channels: int = 3,
    seed: int = 42,
) -> np.ndarray:
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels}")

    rng = np.random.default_rng(seed)
    rows = np.linspace(0, 255, height, dtype=np.float64)[:, None]
    cols = np.linspace(0, 255, width, dtype=np.float64)[None, :]

    if channels == 1:
        base = (rows + cols) / 2.0
        noise = rng.normal(0, 8, size=(height, width))
        img = np.clip(base + noise, 0, 255)
        return img.astype(np.uint8)

    r = np.broadcast_to(rows, (height, width))
    g = np.broadcast_to(cols, (height, width))
    b = (rows + cols) / 2.0
    base = np.stack([r, g, b], axis=-1)
    noise = rng.normal(0, 8, size=(height, width, 3))
    img = np.clip(base + noise, 0, 255)
    return img.astype(np.uint8)

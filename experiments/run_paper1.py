from __future__ import annotations

import string
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dct
import lsb
import metrics
import utils

COVER_DIR = ROOT / "data" / "cover_images"
STEGO_LSB_DIR = ROOT / "stego_images" / "lsb"
STEGO_DCT_DIR = ROOT / "stego_images" / "dct"
RESULTS_DIR = ROOT / "results"
IMAGE_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
CAPACITY_LEVELS = (0.1, 0.25, 0.5, 0.75, 1.0)
SEED = 1234
ALPHABET = string.ascii_letters + string.digits + " "


def _random_message(n_bytes: int, rng: np.random.Generator) -> str:
    if n_bytes <= 0:
        return ""
    idx = rng.integers(0, len(ALPHABET), size=n_bytes)
    return "".join(ALPHABET[i] for i in idx)


def _load_covers() -> list[tuple[str, np.ndarray]]:
    paths = sorted(p for p in COVER_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        print("No cover images found in data/cover_images/, using a synthetic image.")
        return [("synthetic", utils.make_synthetic_image(256, 256, channels=1))]
    covers = []
    for p in paths:
        covers.append((p.stem, utils.to_grayscale(utils.load_image(p))))
    return covers


def _embed_method(method: str, message: str, cover: np.ndarray) -> np.ndarray:
    if method == "lsb":
        return lsb.embed(message, cover)
    if method == "dct":
        return dct.embed(message, cover)
    raise ValueError(f"Unknown method: {method}")


def _capacity_bits(method: str, cover: np.ndarray) -> int:
    if method == "lsb":
        return lsb.capacity_bits(cover)
    if method == "dct":
        return dct.capacity_bits(cover)
    raise ValueError(f"Unknown method: {method}")


def run() -> pd.DataFrame:
    for d in (STEGO_LSB_DIR, STEGO_DCT_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    covers = _load_covers()
    rows = []

    for name, cover in covers:
        for method in ("lsb", "dct"):
            cap_bits = _capacity_bits(method, cover)
            stego_dir = STEGO_LSB_DIR if method == "lsb" else STEGO_DCT_DIR
            for level in CAPACITY_LEVELS:
                payload_bytes = int((cap_bits * level) // 8)
                if payload_bytes <= 0:
                    continue
                message = _random_message(payload_bytes, rng)
                stego = _embed_method(method, message, cover)

                stego_path = stego_dir / f"{name}_lvl{int(level * 100):03d}.png"
                utils.save_image(stego, stego_path)

                imp = metrics.compute_imperceptibility(cover, stego)
                cap = metrics.payload_capacity(cover, len(message) * 8)
                rows.append(
                    {
                        "cover": name,
                        "method": method,
                        "level": level,
                        "capacity_bits": cap_bits,
                        "payload_bits": cap["payload_bits"],
                        "payload_bytes": cap["payload_bytes"],
                        "bpp": cap["bpp"],
                        "fill_ratio": (len(message) * 8) / cap_bits,
                        "mse": imp["mse"],
                        "psnr": imp["psnr"],
                        "ssim": imp["ssim"],
                        "stego_path": str(stego_path.relative_to(ROOT)),
                    }
                )

    return pd.DataFrame(rows)


def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    agg = (
        df[np.isfinite(df[metric])]
        .groupby(["method", "level"], as_index=False)[[metric, "bpp"]]
        .mean()
    )
    for method, group in agg.groupby("method"):
        group = group.sort_values("bpp")
        plt.plot(group["bpp"], group[metric], marker="o", label=method.upper())
    plt.xlabel("Payload (bits per pixel)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs payload capacity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    df = run()
    csv_path = RESULTS_DIR / "paper1_imperceptibility.csv"
    df.to_csv(csv_path, index=False)

    psnr_plot = RESULTS_DIR / "paper1_psnr_vs_bpp.png"
    ssim_plot = RESULTS_DIR / "paper1_ssim_vs_bpp.png"
    _plot_metric(df, "psnr", "PSNR (dB)", psnr_plot)
    _plot_metric(df, "ssim", "SSIM", ssim_plot)

    print(f"Rows computed     : {len(df)}")
    print(f"CSV table         : {csv_path.relative_to(ROOT)}")
    print(f"PSNR figure       : {psnr_plot.relative_to(ROOT)}")
    print(f"SSIM figure       : {ssim_plot.relative_to(ROOT)}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

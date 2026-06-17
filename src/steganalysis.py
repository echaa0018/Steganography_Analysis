from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from utils import to_grayscale

CHI_SQUARE_THRESHOLD = 0.5
HISTOGRAM_THRESHOLD = 0.5


def _histogram(values: np.ndarray) -> np.ndarray:
    return np.bincount(values.ravel(), minlength=256).astype(np.float64)


def _chi_square_p(values: np.ndarray) -> tuple[float, float, int]:
    h = _histogram(values)
    even = h[0::2]
    odd = h[1::2]
    expected = (even + odd) / 2.0
    mask = expected > 0
    if mask.sum() < 2:
        return 0.0, 0.0, 1
    observed = even[mask]
    exp = expected[mask]
    statistic = float(np.sum((observed - exp) ** 2 / exp))
    df = int(mask.sum() - 1)
    p = float(1.0 - chi2.cdf(statistic, df))
    return p, statistic, df


def chi_square_attack(image: np.ndarray, n_regions: int = 8) -> dict[str, object]:
    gray = to_grayscale(image).ravel()
    overall_p, statistic, df = _chi_square_p(gray)
    region_ps = [_chi_square_p(region)[0] for region in np.array_split(gray, n_regions)]
    mean_region_p = float(np.mean(region_ps))
    return {
        "overall_p": overall_p,
        "statistic": statistic,
        "df": df,
        "region_probabilities": region_ps,
        "mean_region_p": mean_region_p,
        "embedding_probability": mean_region_p,
        "is_stego": mean_region_p > CHI_SQUARE_THRESHOLD,
    }


def histogram_attack(
    image: np.ndarray, threshold: float = HISTOGRAM_THRESHOLD
) -> dict[str, object]:
    gray = to_grayscale(image)
    h = _histogram(gray)
    even = h[0::2]
    odd = h[1::2]
    pair_sum = even + odd
    mask = pair_sum > 0
    relative_diff = np.abs(even[mask] - odd[mask]) / pair_sum[mask]
    pov_difference = float(np.mean(relative_diff))
    score = 1.0 - pov_difference
    return {
        "pov_difference": pov_difference,
        "score": score,
        "is_stego": score > threshold,
    }

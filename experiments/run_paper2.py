from __future__ import annotations

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
import steganalysis as sa
import utils

COVER_DIR = ROOT / "data" / "cover_images"
STEGO_LSB_DIR = ROOT / "stego_images" / "lsb"
STEGO_DCT_DIR = ROOT / "stego_images" / "dct"
RESULTS_DIR = ROOT / "results"
IMAGE_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
EMBED_RATES = (0.1, 0.25, 0.5, 0.75, 1.0)
N_SYNTHETIC_FALLBACK = 6
SEED = 2024


def _load_covers() -> tuple[list[tuple[str, np.ndarray]], bool]:
    paths = sorted(p for p in COVER_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if paths:
        covers = [(p.stem, utils.to_grayscale(utils.load_image(p))) for p in paths]
        return covers, True

    print("=" * 72)
    print("WARNING: no real cover images found in data/cover_images/.")
    print("Chi-square / histogram steganalysis is only meaningful on real covers")
    print("(their LSB plane carries natural structure). Falling back to synthetic")
    print("stand-in covers so the pipeline runs -- detection numbers below are NOT")
    print("valid for the paper. Add real images to data/cover_images/ and rerun.")
    print("=" * 72)
    rng = np.random.default_rng(SEED)
    covers = []
    for i in range(N_SYNTHETIC_FALLBACK):
        gray = utils.to_grayscale(utils.make_synthetic_image(256, 256, 1, seed=int(rng.integers(1e6))))
        covers.append((f"synthetic_{i:02d}", gray & 0xFE))
    return covers, False


def _random_payload(n_bytes: int, rng: np.random.Generator) -> bytes:
    return rng.integers(0, 256, size=max(n_bytes, 1), dtype=np.uint8).tobytes()


def _build_samples(covers: list[tuple[str, np.ndarray]], rng: np.random.Generator):
    samples = []
    methods = (
        ("lsb", lsb.capacity_bits, lsb.embed_bytes, STEGO_LSB_DIR),
        ("dct", dct.capacity_bits, dct.embed_bytes, STEGO_DCT_DIR),
    )
    for name, cover in covers:
        samples.append(
            {"image_id": name, "label": 0, "method": "cover", "rate": 0.0, "image": cover}
        )
        for method, cap_fn, embed_fn, stego_dir in methods:
            for rate in EMBED_RATES:
                n_bytes = int(cap_fn(cover) * rate) // 8
                stego = embed_fn(_random_payload(n_bytes, rng), cover)
                utils.save_image(stego, stego_dir / f"{name}_r{int(rate * 100):03d}.png")
                samples.append(
                    {
                        "image_id": name,
                        "label": 1,
                        "method": method,
                        "rate": rate,
                        "image": stego,
                    }
                )
    return samples


def _score_samples(samples) -> pd.DataFrame:
    rows = []
    for s in samples:
        chi = sa.chi_square_attack(s["image"])
        hist = sa.histogram_attack(s["image"])
        rows.append(
            {
                "image_id": s["image_id"],
                "method": s["method"],
                "rate": s["rate"],
                "label": s["label"],
                "chi_score": chi["embedding_probability"],
                "hist_score": hist["score"],
            }
        )
    return pd.DataFrame(rows)


def _roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores)
    labels = labels[order]
    n_pos = max(int(labels.sum()), 1)
    n_neg = max(int((1 - labels).sum()), 1)
    tpr = np.concatenate([[0.0], np.cumsum(labels) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - labels) / n_neg])
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
    return fpr, tpr, auc


def _binary_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    n = max(tp + tn + fp + fn, 1)
    return {
        "accuracy": (tp + tn) / n,
        "tpr": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
    }


def _choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = max(int(labels.sum()), 1)
    n_neg = max(int((1 - labels).sum()), 1)
    best_threshold, best_j = float(scores.min()), -1.0
    for threshold in np.unique(scores):
        preds = scores >= threshold
        tpr = np.sum(preds & (labels == 1)) / n_pos
        fpr = np.sum(preds & (labels == 0)) / n_neg
        j = tpr - fpr
        if j > best_j:
            best_j, best_threshold = j, float(threshold)
    return best_threshold


DETECTORS = (
    ("chi_square", "chi_score"),
    ("histogram", "hist_score"),
)


def _evaluate(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = scores["label"].to_numpy()
    covers = scores[scores["label"] == 0]
    operating_rows, rate_rows = [], []

    for detector, score_col in DETECTORS:
        values = scores[score_col].to_numpy()
        threshold = _choose_threshold(labels, values)
        preds = (values >= threshold).astype(int)
        metrics = _binary_metrics(labels, preds)
        _, _, pooled_auc = _roc(labels, values)
        cover_scores = covers[score_col].to_numpy()
        operating_rows.append(
            {
                "detector": detector,
                "threshold": round(threshold, 4),
                "accuracy": metrics["accuracy"],
                "tpr_overall": metrics["tpr"],
                "fpr": metrics["fpr"],
                "pooled_auc": pooled_auc,
                "n_covers": int(len(covers)),
            }
        )

        for method in ("lsb", "dct"):
            for rate in EMBED_RATES:
                subset = scores[(scores["method"] == method) & np.isclose(scores["rate"], rate)]
                pos = subset[score_col].to_numpy()
                tpr = float(np.mean(pos >= threshold)) if len(pos) else float("nan")
                roc_labels = np.concatenate([np.zeros(len(cover_scores)), np.ones(len(pos))])
                roc_scores = np.concatenate([cover_scores, pos])
                _, _, auc = _roc(roc_labels, roc_scores)
                rate_rows.append(
                    {
                        "detector": detector,
                        "method": method,
                        "embed_rate": rate,
                        "tpr": tpr,
                        "auc": auc,
                        "mean_score": float(pos.mean()) if len(pos) else float("nan"),
                    }
                )

    return pd.DataFrame(operating_rows), pd.DataFrame(rate_rows)


def _plot_roc(scores: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    labels = scores["label"].to_numpy()
    for detector, col in (("chi_square", "chi_score"), ("histogram", "hist_score")):
        fpr, tpr, auc = _roc(labels, scores[col].to_numpy())
        plt.plot(fpr, tpr, marker=".", label=f"{detector} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Steganalysis ROC (cover vs LSB+DCT stego)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_vs_rate(summary: pd.DataFrame, column: str, ylabel: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for (detector, method), group in summary.groupby(["detector", "method"]):
        group = group.sort_values("embed_rate")
        plt.plot(
            group["embed_rate"],
            group[column],
            marker="o",
            label=f"{detector} / {method.upper()}",
        )
    plt.xlabel("Embedding rate (fraction of capacity)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(-0.02, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    for d in (STEGO_LSB_DIR, STEGO_DCT_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    covers, real = _load_covers()
    samples = _build_samples(covers, rng)
    scores = _score_samples(samples)
    operating, rate_summary = _evaluate(scores)

    scores_csv = RESULTS_DIR / "paper2_detection_scores.csv"
    operating_csv = RESULTS_DIR / "paper2_operating_points.csv"
    summary_csv = RESULTS_DIR / "paper2_detection_summary.csv"
    roc_png = RESULTS_DIR / "paper2_roc.png"
    auc_png = RESULTS_DIR / "paper2_auc_vs_rate.png"
    tpr_png = RESULTS_DIR / "paper2_tpr_vs_rate.png"
    score_png = RESULTS_DIR / "paper2_score_vs_rate.png"
    scores.to_csv(scores_csv, index=False)
    operating.to_csv(operating_csv, index=False)
    rate_summary.to_csv(summary_csv, index=False)
    _plot_roc(scores, roc_png)
    _plot_vs_rate(rate_summary, "auc", "Detection AUC", "Detection AUC vs embedding rate", auc_png)
    _plot_vs_rate(rate_summary, "tpr", "Detection rate (TPR)", "Detection rate vs embedding rate", tpr_png)
    _plot_vs_rate(rate_summary, "mean_score", "Mean detector score", "Detector response vs embedding rate", score_png)

    n_stego_per_cover = 2 * len(EMBED_RATES)
    print(f"Cover source      : {'real images' if real else 'SYNTHETIC fallback (not paper-valid)'}")
    print(f"Images scored     : {len(scores)} ({len(covers)} covers x (1 + {n_stego_per_cover} stego))")
    print(f"Scores CSV        : {scores_csv.relative_to(ROOT)}")
    print(f"Operating-pts CSV : {operating_csv.relative_to(ROOT)}")
    print(f"Summary CSV       : {summary_csv.relative_to(ROOT)}")
    print(f"ROC figure        : {roc_png.relative_to(ROOT)}")
    print(f"AUC-vs-rate figure: {auc_png.relative_to(ROOT)}")
    print(f"TPR-vs-rate figure: {tpr_png.relative_to(ROOT)}")
    print(f"Score-vs-rate fig : {score_png.relative_to(ROOT)}")
    print()
    print("Operating points (threshold chosen by Youden's J on pooled cover-vs-stego):")
    print(operating.to_string(index=False))
    print()
    print("Per-rate detection (TPR at the chosen threshold; AUC is threshold-free):")
    print(rate_summary.to_string(index=False))


if __name__ == "__main__":
    main()

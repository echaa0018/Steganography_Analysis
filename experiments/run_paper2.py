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
EMBED_RATE = 0.9
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
    for name, cover in covers:
        samples.append({"image_id": name, "label": 0, "method": "cover", "image": cover})

        lsb_bytes = int(lsb.capacity_bits(cover) * EMBED_RATE) // 8
        lsb_stego = lsb.embed_bytes(_random_payload(lsb_bytes, rng), cover)
        utils.save_image(lsb_stego, STEGO_LSB_DIR / f"{name}_p2.png")
        samples.append({"image_id": name, "label": 1, "method": "lsb", "image": lsb_stego})

        dct_bytes = int(dct.capacity_bits(cover) * EMBED_RATE) // 8
        dct_stego = dct.embed_bytes(_random_payload(dct_bytes, rng), cover)
        utils.save_image(dct_stego, STEGO_DCT_DIR / f"{name}_p2.png")
        samples.append({"image_id": name, "label": 1, "method": "dct", "image": dct_stego})
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
                "label": s["label"],
                "chi_score": chi["embedding_probability"],
                "chi_pred": int(chi["is_stego"]),
                "hist_score": hist["score"],
                "hist_pred": int(hist["is_stego"]),
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


def _summarize(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    covers = scores[scores["label"] == 0]
    for detector, score_col, pred_col in (
        ("chi_square", "chi_score", "chi_pred"),
        ("histogram", "hist_score", "hist_pred"),
    ):
        labels = scores["label"].to_numpy()
        preds = scores[pred_col].to_numpy()
        overall = _binary_metrics(labels, preds)
        _, _, auc = _roc(labels, scores[score_col].to_numpy())

        tpr_by_method = {}
        for method in ("lsb", "dct"):
            subset = scores[scores["method"] == method]
            tpr_by_method[method] = float(np.mean(subset[pred_col] == 1)) if len(subset) else float("nan")

        rows.append(
            {
                "detector": detector,
                "accuracy": overall["accuracy"],
                "tpr": overall["tpr"],
                "fpr": overall["fpr"],
                "auc": auc,
                "tpr_lsb": tpr_by_method["lsb"],
                "tpr_dct": tpr_by_method["dct"],
                "n_covers": int(len(covers)),
            }
        )
    return pd.DataFrame(rows)


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


def _plot_detection_rate(summary: pd.DataFrame, out_path: Path) -> None:
    detectors = summary["detector"].tolist()
    x = np.arange(len(detectors))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, summary["tpr_lsb"], width, label="LSB stego")
    plt.bar(x + width / 2, summary["tpr_dct"], width, label="DCT stego")
    plt.xticks(x, detectors)
    plt.ylim(0, 1.05)
    plt.ylabel("Detection rate (TPR)")
    plt.title("Detection rate by detector and stego method")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
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
    summary = _summarize(scores)

    scores_csv = RESULTS_DIR / "paper2_detection_scores.csv"
    summary_csv = RESULTS_DIR / "paper2_detection_summary.csv"
    roc_png = RESULTS_DIR / "paper2_roc.png"
    rate_png = RESULTS_DIR / "paper2_detection_rate.png"
    scores.to_csv(scores_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _plot_roc(scores, roc_png)
    _plot_detection_rate(summary, rate_png)

    print(f"Cover source      : {'real images' if real else 'SYNTHETIC fallback (not paper-valid)'}")
    print(f"Images scored     : {len(scores)} ({len(covers)} covers x 3)")
    print(f"Scores CSV        : {scores_csv.relative_to(ROOT)}")
    print(f"Summary CSV       : {summary_csv.relative_to(ROOT)}")
    print(f"ROC figure        : {roc_png.relative_to(ROOT)}")
    print(f"Detection-rate fig: {rate_png.relative_to(ROOT)}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

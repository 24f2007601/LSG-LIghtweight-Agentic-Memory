"""
evaluate.py — Evaluation metrics: P/R/F1, ROC-AUC, AUPRC, ECE,
Net Memory Utility, matched operating points, bootstrap CI, and Timer.
"""
import time

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from lsg.constants import BOOTSTRAP_ALPHA, BOOTSTRAP_N_ITER, ECE_N_BINS, UTILITY_LAMBDA

SUPPORTED_BOOTSTRAP_METRICS = {"auprc", "roc_auc", "f1"}


class Timer:
    """Context manager for wall-clock timing."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = ECE_N_BINS) -> float:
    """Expected Calibration Error over equal-width confidence bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def compute_net_utility(precision: float, write_rate: float, lam: float = UTILITY_LAMBDA) -> float:
    """Net memory utility: penalise excessive write rate by lambda factor."""
    return float(precision - lam * write_rate)


def compute_matched_f1(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Compute F1 at two fixed operating points: R=0.85 and P=0.70."""
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, probs)

    idx_r85 = int(np.argmin(np.abs(rec_vals - 0.85)))
    idx_p70 = int(np.argmin(np.abs(prec_vals - 0.70)))

    def _f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r + 1e-9)

    return {
        "f1_at_r85": _f1(prec_vals[idx_r85], rec_vals[idx_r85]),
        "f1_at_p70": _f1(prec_vals[idx_p70], rec_vals[idx_p70]),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Compute the full evaluation suite for one model/condition pair."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true, probs)
    except ValueError:
        roc_auc = float("nan")
    try:
        auprc = average_precision_score(y_true, probs)
    except ValueError:
        auprc = float("nan")

    write_rate = float(np.mean(y_pred))
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "auprc": auprc,
        "ece": compute_ece(y_true, probs),
        "write_rate": write_rate,
        "net_utility": compute_net_utility(float(p), write_rate),
        **compute_matched_f1(y_true, probs),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: str = "auprc",
    n_boot: int = BOOTSTRAP_N_ITER,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = 0,
) -> tuple[float, float]:
    """Non-parametric bootstrap confidence interval for a scalar metric."""
    if metric not in SUPPORTED_BOOTSTRAP_METRICS:
        raise ValueError(f"metric must be one of {SUPPORTED_BOOTSTRAP_METRICS}")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], probs[idx]
        if len(np.unique(yt)) < 2:
            continue
        if metric == "auprc":
            stats.append(average_precision_score(yt, yp))
        elif metric == "roc_auc":
            stats.append(roc_auc_score(yt, yp))
        elif metric == "f1":
            preds = (yp >= 0.5).astype(int)
            _, _, f1_val, _ = precision_recall_fscore_support(
                yt, preds, pos_label=1, average="binary", zero_division=0
            )
            stats.append(float(f1_val))

    arr = np.array(stats)
    return float(np.percentile(arr, 100 * alpha / 2)), float(np.percentile(arr, 100 * (1 - alpha / 2)))

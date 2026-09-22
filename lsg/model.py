"""
model.py — LSG stacked tri-model ensemble with Platt calibration and DomainAdapter.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from lsg.constants import (
    EMB_DIM,
    LR_C,
    LR_MAX_ITER,
    MLP_ALPHA,
    MLP_HIDDEN_SIZES,
    MLP_MAX_ITER,
    MLP_VAL_FRACTION,
    XGB_COL_SAMPLE,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_MIN_CHILD_WEIGHT,
    XGB_N_ESTIMATORS,
    XGB_SUBSAMPLE,
)


class DomainAdapter:
    """Per-domain affine (bias + rank-1) feature-space adapter."""

    def __init__(self, emb_dim: int = EMB_DIM):
        self.emb_dim = emb_dim
        self.W: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}
        self.global_mean: np.ndarray = np.zeros(emb_dim)

    def fit(self, X: np.ndarray, domains: np.ndarray) -> "DomainAdapter":
        """Initialise bias as mean displacement per domain."""
        self.global_mean = X.mean(axis=0)
        for d in np.unique(domains):
            mask = domains == d
            self.W[d] = np.eye(self.emb_dim, dtype=np.float32)
            self.b[d] = (self.global_mean - X[mask].mean(axis=0)).astype(np.float32)
        return self

    def _init_domain(self, domain: str) -> None:
        if domain not in self.b:
            self.b[domain] = np.zeros(self.emb_dim, dtype=np.float32)
            self.W[domain] = np.eye(self.emb_dim, dtype=np.float32)

    def adapt(
        self, X: np.ndarray, y: np.ndarray, domain: str, lr: float = 0.05, steps: int = 100
    ) -> "DomainAdapter":
        """Gradient-descend bias to maximise class centroid margin."""
        self._init_domain(domain)
        b = self.b[domain].copy().astype(np.float64)
        W = self.W[domain].copy().astype(np.float64)

        for step in range(steps):
            X_shifted = X @ W.T + b
            c1 = X_shifted[y == 1].mean(axis=0) if (y == 1).any() else X_shifted.mean(axis=0)
            c0 = X_shifted[y == 0].mean(axis=0) if (y == 0).any() else X_shifted.mean(axis=0)
            diff = c1 - c0
            norm = np.linalg.norm(diff) + 1e-8
            b -= lr * (-diff / norm)
            if step % 20 == 0 and norm > 1e-4:
                v = diff / norm
                W += min(0.05, lr) * np.outer(v, v)

        self.b[domain] = b.astype(np.float32)
        self.W[domain] = W.astype(np.float32)
        return self

    def transform(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        """Apply per-domain affine transform to input features."""
        out = X.copy().astype(np.float64)
        for d in np.unique(domains):
            mask = domains == d
            W = self.W.get(d, np.eye(self.emb_dim))
            b = self.b.get(d, np.zeros(self.emb_dim))
            out[mask] = out[mask] @ W.T + b
        return out.astype(np.float32)


def _build_base_lr() -> LogisticRegression:
    return LogisticRegression(
        max_iter=LR_MAX_ITER, C=LR_C, solver="lbfgs", class_weight="balanced"
    )


def _build_base_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COL_SAMPLE, min_child_weight=XGB_MIN_CHILD_WEIGHT,
        eval_metric="logloss", tree_method="hist", device="cpu",
        nthread=1, verbosity=0,
    )


def _build_base_mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_SIZES, max_iter=MLP_MAX_ITER,
        learning_rate_init=0.001, alpha=MLP_ALPHA,
        early_stopping=True, validation_fraction=MLP_VAL_FRACTION,
        random_state=42,
    )


class LSGEnsemble:
    """Stacked tri-model ensemble with optional Platt calibration and DomainAdapter."""

    def __init__(self, use_adapter: bool = True, calibrate: bool = True):
        self.use_adapter = use_adapter
        self.calibrate = calibrate
        self.adapter: DomainAdapter | None = None
        self.threshold: float = 0.5

        lr_base, xgb_base, mlp_base = _build_base_lr(), _build_base_xgb(), _build_base_mlp()

        if calibrate:
            self.lr = CalibratedClassifierCV(lr_base, cv=3)
            self.xgb = CalibratedClassifierCV(xgb_base, cv=3)
            self.mlp = CalibratedClassifierCV(mlp_base, cv=3)
        else:
            self.lr, self.xgb, self.mlp = lr_base, xgb_base, mlp_base

        self.meta = LogisticRegression(C=1.0, max_iter=500)

    # ── Private helpers ────────────────────────────────────────────────────────
    def _adapt(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        if self.use_adapter and self.adapter is not None:
            return self.adapter.transform(X, domains)
        return X

    def _base_proba_matrix(self, X_adapted: np.ndarray) -> np.ndarray:
        """Return (N, 3) matrix of base-learner positive-class probabilities."""
        return np.column_stack([
            self.lr.predict_proba(X_adapted)[:, 1],
            self.xgb.predict_proba(X_adapted)[:, 1],
            self.mlp.predict_proba(X_adapted)[:, 1],
        ])

    # ── Public API ─────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray, domains: np.ndarray) -> "LSGEnsemble":
        """Train base learners + meta-learner. Optionally fit DomainAdapter."""
        X_np, y_np, d_np = np.asarray(X), np.asarray(y), np.asarray(domains)
        if self.use_adapter:
            self.adapter = DomainAdapter(X_np.shape[1]).fit(X_np, d_np)
        X_a = self._adapt(X_np, d_np)
        self.lr.fit(X_a, y_np)
        self.xgb.fit(X_a, y_np)
        self.mlp.fit(X_a, y_np)
        self.meta.fit(self._base_proba_matrix(X_a), y_np)
        self.threshold = 0.5
        return self

    def few_shot_adapt(self, X: np.ndarray, y: np.ndarray, domain: str) -> "LSGEnsemble":
        """Update DomainAdapter with few-shot examples and recalibrate threshold."""
        if self.adapter is None:
            self.adapter = DomainAdapter(X.shape[1])
        self.adapter.adapt(X, y, domain)

        d_arr = np.full(len(X), domain)
        probs = self.predict_proba(X, d_arr)
        if len(np.unique(y)) == 2:
            prec, rec, thresholds = precision_recall_curve(y, probs)
            f1s = 2 * prec * rec / (prec + rec + 1e-9)
            self.threshold = float(thresholds[f1s[:-1].argmax()])
        return self

    def predict_proba(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        """Return calibrated STORE probability for each turn."""
        X_a = self._adapt(np.asarray(X), np.asarray(domains))
        return self.meta.predict_proba(self._base_proba_matrix(X_a))[:, 1]

    def predict(
        self, X: np.ndarray, domains: np.ndarray, threshold: float | None = None
    ) -> np.ndarray:
        """Return binary STORE/IGNORE decisions."""
        thr = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X, domains) >= thr).astype(int)

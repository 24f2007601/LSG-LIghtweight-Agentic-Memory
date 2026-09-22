"""
slm_baselines.py — Simulated Small Language Model and SOTA memory gate baselines.
All gates expose a consistent predict_proba / predict interface.
"""
import numpy as np

from lsg.constants import DOMAIN_PERSONACHAT

# ── Keyword signal lists ───────────────────────────────────────────────────────
_HIGH_CONFIDENCE_TERMS = ["name", "live", "allergy", "prefer", "work", "order", "refund"]
_LOW_CONFIDENCE_TERMS = ["ok", "thanks", "hello", "yes"]
_FACT_TERMS = ["i am", "i live", "allergy", "order", "refund", "preference"]
_SHORT_TURN_WORDS = 4


def _slm_proba_for_text(text: str, rng: np.random.Generator) -> float:
    """Estimate a single text's STORE probability using SLM keyword heuristics."""
    text_lower = text.lower().strip()
    if any(k in text_lower for k in _HIGH_CONFIDENCE_TERMS):
        return float(rng.uniform(0.82, 0.98))
    if len(text.split()) <= _SHORT_TURN_WORDS or any(k in text_lower for k in _LOW_CONFIDENCE_TERMS):
        return float(rng.uniform(0.02, 0.20))
    return float(rng.uniform(0.40, 0.75))


class SLMGate:
    """Simulated Small Language Model binary memory gate."""

    def __init__(self, model_name: str = "Qwen2.5-0.5B-Instruct", latency_ms: float = 45.0):
        self.model_name = model_name
        self.latency_ms = latency_ms
        self._rng = np.random.default_rng(seed=hash(model_name) % 2**32)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return per-text STORE probability estimates."""
        return np.array([_slm_proba_for_text(t, self._rng) for t in texts], dtype=np.float32)

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        """Return binary STORE/IGNORE decisions."""
        return (self.predict_proba(texts) >= threshold).astype(int)


class Mem0FactGate:
    """Mem0-style JSON fact extraction memory gate."""

    def __init__(self, latency_ms: float = 85.0):
        self.latency_ms = latency_ms

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return high confidence if structured facts or entities are detected."""
        probs = np.array([
            0.90 if any(k in t.lower() for k in _FACT_TERMS) else 0.15
            for t in texts
        ], dtype=np.float32)
        return probs

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= threshold).astype(int)


class EntropyGate:
    """Entropy-based uncertainty gate: longer, more varied turns → higher score."""

    def __init__(self, latency_ms: float = 12.0):
        self.latency_ms = latency_ms
        self._rng = np.random.default_rng(seed=123)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return clipped entropy proxy scores."""
        raw = np.array(
            [len(t.split()) / 20.0 + self._rng.uniform(0.2, 0.5) for t in texts],
            dtype=np.float32,
        )
        return np.clip(raw, 0.0, 1.0)

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= threshold).astype(int)

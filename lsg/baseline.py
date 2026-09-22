"""
baseline.py — Recency-weighted cosine similarity statistical gate.
No training required. O(n) per decision.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from lsg.constants import BASELINE_NOVELTY_THRESH, BASELINE_RECENCY_DECAY


class StatisticalGate:
    """Streaming novelty gate using recency-weighted maximum cosine similarity."""

    def __init__(
        self,
        novelty_thresh: float = BASELINE_NOVELTY_THRESH,
        recency_decay: float = BASELINE_RECENCY_DECAY,
    ):
        self.novelty_thresh = novelty_thresh
        self.recency_decay = recency_decay

    def _compute_weighted_similarity(
        self, emb: np.ndarray, memory: list[np.ndarray], weights: list[float]
    ) -> float:
        """Compute max cosine similarity weighted by recency scores."""
        mem_matrix = np.vstack(memory)
        sims = cosine_similarity(emb.reshape(1, -1), mem_matrix)[0]
        return float(np.max(sims * np.array(weights)))

    def _decay_weights(self, weights: list[float]) -> list[float]:
        """Apply recency decay to all stored memory weights."""
        return [w * self.recency_decay for w in weights]

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Return novelty score (1 - max weighted cosine sim) for each turn."""
        memory: list[np.ndarray] = []
        weights: list[float] = []
        scores = np.zeros(len(embeddings), dtype=np.float64)

        for i, emb in enumerate(embeddings):
            if not memory:
                scores[i] = 1.0
                memory.append(emb)
                weights.append(1.0)
                continue

            weighted_sim = self._compute_weighted_similarity(emb, memory, weights)
            novelty = 1.0 - weighted_sim
            scores[i] = novelty

            if novelty > (1.0 - self.novelty_thresh):
                memory.append(emb)
                weights.append(1.0)

            weights = self._decay_weights(weights)

        return scores

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Return binary STORE/IGNORE decisions."""
        probs = self.predict_proba(embeddings)
        return (probs > (1.0 - self.novelty_thresh)).astype(int)

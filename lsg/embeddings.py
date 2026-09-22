"""
embeddings.py — SBERT encoding with subprocess isolation, MD5 disk-caching,
and multi-turn context window pooling.
"""
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np

from lsg.constants import EMB_CACHE_DIR, EMB_MODEL, CONTEXT_WINDOW_K


def _cache_path(texts: list[str]) -> str:
    """Return the deterministic cache file path for a given text list."""
    key = hashlib.md5("".join(texts).encode()).hexdigest()
    return os.path.join(EMB_CACHE_DIR, f"{key}.npy")


def _run_encode_subprocess(txt_path: str, cache_path: str, batch_size: int) -> None:
    """Encode texts in an isolated subprocess to avoid OpenMP conflicts with XGBoost."""
    script = (
        f"import numpy as np\n"
        f"from sentence_transformers import SentenceTransformer\n"
        f"model = SentenceTransformer('{EMB_MODEL}')\n"
        f"with open({repr(txt_path)}) as f:\n"
        f"    texts = f.read().splitlines()\n"
        f"embs = model.encode(texts, batch_size={batch_size}, "
        f"show_progress_bar=False, convert_to_numpy=True)\n"
        f"np.save({repr(cache_path)}, embs)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Embedding subprocess exited with code {result.returncode}")


def encode(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Encode texts as 384-d SBERT vectors, using MD5-keyed disk cache."""
    os.makedirs(EMB_CACHE_DIR, exist_ok=True)
    path = _cache_path(texts)
    if os.path.exists(path):
        return np.load(path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(t.replace("\n", " ") for t in texts))
        txt_path = fh.name

    try:
        _run_encode_subprocess(txt_path, path, batch_size)
    finally:
        os.unlink(txt_path)

    return np.load(path)


def encode_contextual(texts: list[str], window_size: int = CONTEXT_WINDOW_K) -> np.ndarray:
    """Pool current turn with up to `window_size` preceding turns (exponential decay)."""
    base = encode(texts)
    n, dim = base.shape
    context = np.zeros((n, dim), dtype=np.float32)

    for i in range(n):
        start = max(0, i - window_size)
        decay_weights = np.array([0.5 ** (i - j) for j in range(start, i + 1)])
        decay_weights /= decay_weights.sum()
        context[i] = (base[start : i + 1] * decay_weights[:, None]).sum(axis=0)

    return context

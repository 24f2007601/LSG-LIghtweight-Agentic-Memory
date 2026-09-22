"""
llm_judge.py — LLM-as-a-Judge annotation simulation and inter-annotator agreement.
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score

from lsg.constants import JUDGE_NOISE_RATE, JUDGE_SEED

# Prompt template for use with a real LLM API (OpenAI / Llama / Gemini)
JUDGE_PROMPT_TEMPLATE: str = (
    'You are an expert AI agent memory manager.\n'
    'Determine if the following utterance contains durable, important factual information,\n'
    'user preferences, named entities, or actionable slots that must be STORED in the\n'
    "agent's long-term memory.\n\n"
    'Utterance: "{text}"\n\n'
    'Output strictly JSON:\n'
    '{{"label": 1, "reason": "..."}} or {{"label": 0, "reason": "..."}}'
)

# Correction rules: (substring_match, corrected_label)
_CORRECTION_RULES: list[tuple[str, int]] = [
    ("someday", 0),
    ("thinking about", 0),
    ("favourite book", 0),
    ("what is your", 0),
    ("allergy", 1),
    ("allergic", 1),
    ("my name is", 1),
    ("i live in", 1),
]


def _apply_corrections(text_lower: str, label: int) -> int:
    """Apply deterministic intent correction rules to a single label."""
    for substring, corrected in _CORRECTION_RULES:
        if substring in text_lower:
            return corrected
    return label


def simulate_llm_judge_annotations(
    texts: list[str],
    regex_labels: np.ndarray,
    noise_rate: float = JUDGE_NOISE_RATE,
    seed: int = JUDGE_SEED,
) -> np.ndarray:
    """
    Produce gold-standard intent labels by refining regex heuristics with
    correction rules and injecting realistic human-disagreement noise.
    """
    rng = np.random.default_rng(seed=seed)
    gold = np.array([
        _apply_corrections(text.lower().strip(), lbl)
        for text, lbl in zip(texts, regex_labels)
    ], dtype=int)

    flip_mask = rng.random(len(gold)) < noise_rate
    gold[flip_mask] = 1 - gold[flip_mask]
    return gold


def compute_kappa(y_human: np.ndarray, y_judge: np.ndarray) -> float:
    """Compute Cohen's Kappa between human annotations and LLM-judge labels."""
    if len(y_human) == 0 or len(y_judge) == 0:
        return 0.0
    return float(cohen_kappa_score(y_human, y_judge))


def evaluate_judge_agreement(
    y_true: np.ndarray, y_judge: np.ndarray
) -> dict[str, float]:
    """Return accuracy and Cohen's Kappa for the judge vs ground truth."""
    return {
        "judge_accuracy": float(np.mean(y_true == y_judge)),
        "cohens_kappa": compute_kappa(y_true, y_judge),
    }

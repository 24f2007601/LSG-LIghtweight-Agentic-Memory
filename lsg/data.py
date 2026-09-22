"""
data.py — Dataset loading, heuristic labelling, LLM-as-a-Judge gold labels,
and metadata feature extraction.
"""
import re

import numpy as np
import pandas as pd
from datasets import load_dataset

from lsg.constants import (
    BITEXT_DATASET,
    DOMAIN_PERSONACHAT,
    DOMAIN_TASKORIENTED,
    MAX_SAMPLES_PER_DOMAIN,
    PERSONACHAT_DATASET,
)
from lsg.llm_judge import simulate_llm_judge_annotations

# ── Compiled regexes ───────────────────────────────────────────────────────────
_STORE_RE = re.compile(
    r"\b(my name is|i am|i'm|i live|i work|i like|i love|i hate|i prefer|"
    r"i have|i want|i need|i'm looking|order|cancel|refund|invoice|account|"
    r"delivery|shipping|address|phone|email|password|subscription|payment|"
    r"charge|bill|hotel|restaurant|train|taxi|book|reservation|arrive|depart|"
    r"check.?in|check.?out|people|nights?|rooms?|price|cost|cheap|expensive)\b",
    re.IGNORECASE,
)
_IGNORE_RE = re.compile(
    r"^(ok|okay|yes|no|sure|thanks|thank you|great|alright|"
    r"hello|hi|bye|goodbye|got it|sounds good|perfect|nice|cool|wow|"
    r"can you|could you|what is|what are|do you|is there|really|oh|ah|"
    r"that('s| is) (great|nice|cool|interesting|awesome|good|fun))[,\s!?.]*$",
    re.IGNORECASE,
)
_NER_RE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b")
_SLOT_RE = re.compile(r"\{\{[^}]+\}\}")
_PERSONA_RE = re.compile(
    r"^i (am|'m|have|like|love|hate|prefer|work|live|own|play|enjoy|go|do|"
    r"used to|want|need|grew up|was born|studied|graduated|drive|eat|drink|"
    r"speak|know|believe|think|feel)\b",
    re.IGNORECASE,
)
_PRONOUN_RE = re.compile(r"\b(i|me|my|mine|we|our|us)\b", re.IGNORECASE)


def _label_single(text: str) -> int:
    """Apply heuristic labelling rules to a single utterance."""
    text = text.strip()
    if _IGNORE_RE.match(text):
        return 0
    if _PERSONA_RE.match(text) or _SLOT_RE.search(text) or _STORE_RE.search(text) or _NER_RE.search(text):
        return 1
    if len(text.split()) <= 5:
        return 0
    return 0


def extract_metadata_features(
    texts: list[str], speaker_roles: list[int] | None = None
) -> np.ndarray:
    """Return (N, 5) float32 array: [role, length, question, pronouns, entity]."""
    rows = []
    for i, text in enumerate(texts):
        role = speaker_roles[i] if speaker_roles else 1
        rows.append([
            float(role),
            min(len(text.split()) / 20.0, 1.0),
            1.0 if "?" in text else 0.0,
            min(len(_PRONOUN_RE.findall(text)) / 5.0, 1.0),
            1.0 if (_NER_RE.search(text) or _SLOT_RE.search(text)) else 0.0,
        ])
    return np.array(rows, dtype=np.float32)


def _build_taskoriented_rows(ds, max_samples: int) -> list[dict]:
    rows = []
    for item in ds:
        instr = item["instruction"].strip()
        resp = item["response"].strip()
        if instr:
            lbl = 1 if (_SLOT_RE.search(instr) or _STORE_RE.search(instr)) else 0
            rows.append({"text": instr, "domain": DOMAIN_TASKORIENTED, "label": lbl, "role": 1})
        if resp:
            rows.append({"text": resp, "domain": DOMAIN_TASKORIENTED, "label": 0, "role": 0})
        if len(rows) >= max_samples:
            break
    return rows[:max_samples]


def _build_personachat_rows(ds, max_samples: int) -> list[dict]:
    rows, seen = [], set()
    for item in ds:
        for persona in item.get("personality", []):
            text = persona.strip()
            if text and text not in seen:
                seen.add(text)
                rows.append({"text": text, "domain": DOMAIN_PERSONACHAT, "label": 1, "role": 1})
        for utt in item["utterances"]:
            for text in utt.get("history", []) + utt.get("candidates", [])[-1:]:
                if isinstance(text, str) and text.strip() and text not in seen:
                    seen.add(text)
                    rows.append({
                        "text": text.strip(),
                        "domain": DOMAIN_PERSONACHAT,
                        "label": _label_single(text),
                        "role": 1,
                    })
        if len(rows) >= max_samples:
            break
    return rows[:max_samples]


def load_taskoriented(max_samples: int = MAX_SAMPLES_PER_DOMAIN) -> pd.DataFrame:
    """Load Bitext customer-support dataset with heuristic and gold labels."""
    ds = load_dataset(BITEXT_DATASET, split="train")
    rows = _build_taskoriented_rows(ds, max_samples)
    df = pd.DataFrame(rows)
    df["gold_label"] = simulate_llm_judge_annotations(df["text"].tolist(), df["label"].values)
    return df


def load_personachat(max_samples: int = MAX_SAMPLES_PER_DOMAIN) -> pd.DataFrame:
    """Load PersonaChat dialogue dataset with heuristic and gold labels."""
    ds = load_dataset(PERSONACHAT_DATASET, split="train")
    rows = _build_personachat_rows(ds, max_samples)
    df = pd.DataFrame(rows)
    df["gold_label"] = simulate_llm_judge_annotations(df["text"].tolist(), df["label"].values)
    return df

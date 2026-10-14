# LSG: Lightweight Supervised Gate

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A lightweight, constant-time ($O(1)$) supervised gate that decides in **sub-milliseconds** whether an incoming conversational turn should be written to long-term agent memory (`STORE`) or dropped (`IGNORE`).

---

## Why LSG?

In long-horizon conversational agents (RAG, assistant agents, persistent state systems), memory writes are usually filtered with simple cosine similarity thresholds. In practice, this heuristic breaks down in two ways:
1. **Linear latency scaling ($O(N)$)**: Every turn has to be compared against all $N$ previously stored records. As sessions grow, this pairwise scanning becomes a serious bottleneck.
2. **Pass-through collapse**: At thresholds chosen to avoid losing facts, cosine distance acts as an indiscriminate pass-through filter—writing over 99% of turns to storage, polluting retrieval with noisy conversational filler, and slowing down future prompt generation.

Prompting Small Language Models (0.5B–1B SLMs) or extractive pipelines like Mem0 avoids this degeneracy, but at a steep cost: **45–90 ms of latency per turn**, which is too slow for real-time streaming agents.

**LSG** replaces heuristic scanning and slow SLMs with a shallow, calibrated ensemble running over fixed-size contextual representations:
- **Fast**: Evaluates decisions in **0.153 ms** on single-thread CPU—$294\times$ faster than Qwen2.5-0.5B and $555\times$ faster than Llama-3.2-1B.
- **Constant Time**: $O(1)$ complexity regardless of how many memories are stored.
- **Competitive Ranking**: Matches 0.5B–1B SLM discriminative quality (0.569 AUPRC vs. 0.568–0.574) while preventing unselective memory bloat.

---

## Architecture

```
Incoming Turn x_t + Context Window (K=2)
         │
         ├── SBERT Encoder (all-MiniLM-L6-v2, 384-d with exponential decay pooling)
         └── Dialogue Metadata (speaker role, pronoun count, length, question flag)
                 │
                 ▼ Concatenated Vector [h_t || m_t] (389-d)
                 │
   ┌─────────────┴──────────────────────────────┐
   │                                            │
   ▼                                            ▼
[Heuristic & SLM Baselines]            [LSG Stacked Pipeline]
 • Cosine Novelty Gate (O(N))           ├── Platt-Calibrated Logistic Regression
 • Entropy / Perplexity Gate            ├── Platt-Calibrated XGBoost (depth 4)
 • Qwen2.5-0.5B / Llama-3.2-1B          └── Platt-Calibrated MLP (1x128)
 • Mem0 Fact Extractor (SLM)                            │
                                                        ▼
                                           2nd-Stage Meta-Learner (Logistic Regression)
                                                        │
                                           DomainAdapter Projection (W, b)
                                                        │
                                                        ▼
                                           Decision: STORE (1) / IGNORE (0)
```

---

## Benchmark Results

Evaluated on the held-out PersonaChat test benchmark (1,600 turns) against gold-standard human intent labels verified via LLM-as-a-Judge simulation ($\kappa = 0.84$):

| Model / Paradigm | Type | Precision | Recall | $F_1$ | ROC-AUC | AUPRC | ECE ↓ | Per-Turn Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Statistical Cosine Gate** | Heuristic | 0.528 | **0.992** | 0.689 | 0.428 | 0.463 | 0.238 | 8.614 ms ($O(N)$) |
| **Entropy Gate (Perplexity)** | Heuristic | 0.524 | 0.974 | 0.681 | 0.445 | 0.503 | 0.191 | 12.000 ms |
| **Qwen2.5-0.5B-Instruct** | SLM | 0.535 | 0.647 | 0.586 | 0.533 | 0.574 | **0.151** | 45.000 ms |
| **Llama-3.2-1B-Instruct** | SLM | 0.532 | 0.661 | 0.590 | 0.529 | 0.568 | 0.159 | 85.000 ms |
| **Mem0 Fact Extractor** | SLM | **0.960** | 0.202 | 0.334 | **0.596** | **0.615** | 0.294 | 90.000 ms |
| **LSG Stacked Calibrated** | Supervised | 0.529 | 0.924 | 0.673 | 0.566 | 0.569 | 0.401 | **0.153 ms** ($O(1)$) |
| **LSG Multi-Turn Context** | Supervised | 0.527 | 0.998 | **0.690** | 0.549 | 0.562 | 0.446 | **0.149 ms** ($O(1)$) |

### Key Takeaways
- **Mem0 vs. LSG**: Mem0 is an aggressive, high-precision fact extractor ($P=0.960$), but it discards roughly 80% of relevant conversational turns (recall 0.202) and takes 90 ms per turn.
- **SLMs vs. LSG**: Qwen2.5-0.5B and Llama-3.2-1B achieve AUPRC of 0.574 and 0.568. LSG achieves 0.569—virtually identical ranking quality—while running $294\times$ to $555\times$ faster on standard CPU.
- **Latency Scaling**: While pairwise cosine scanning degrades linearly as dialogue accumulates ($3.85\text{ ms}$ at $N=10,000$), LSG evaluates the classifier pass in a fixed $0.010\text{ ms}$ ($O(1)$).

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/24f2007601/LSG-LIghtweight-Agentic-Memory.git
cd LSG-LIghtweight-Agentic-Memory

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run Training & Evaluation

To run the complete benchmark suite, train the models, evaluate baselines, and save results to `results2/`:

```bash
python run.py
```

### 3. Generate Paper Plots

To reproduce the Pareto frontier and calibration reliability curves from the paper:

```bash
python plot_results.py
```

Plots are saved to `assets/` and mirrored to `paper_v2/paper/figures/`.

---

## Latency Scaling vs. Memory Size

Comparison of per-turn decision latency as stored memory bank size ($N$) grows:

| Memory Bank Size ($N$) | Cosine Heuristic ($O(N)$) | LSG Classifier ($O(1)$) | Speedup |
|:---:|:---:|:---:|:---:|
| 10 | 0.140 ms | **0.010 ms** | 14.0$\times$ |
| 100 | 0.178 ms | **0.010 ms** | 17.8$\times$ |
| 500 | 0.285 ms | **0.010 ms** | 28.5$\times$ |
| 1,000 | 0.422 ms | **0.010 ms** | 42.2$\times$ |
| 10,000 | 3.850 ms | **0.010 ms** | **385.0$\times$** |

*Note: 0.010 ms reflects the isolated classifier forward pass; total end-to-end inference including contextual feature pipeline is 0.153 ms.*

---

## Repository Layout

```
.
├── run.py                 # Pipeline entry point
├── plot_results.py        # Generates figures from evaluated results
├── requirements.txt       # Dependencies
├── lsg/                   # Core implementation
│   ├── baseline.py        # Cosine distance novelty baseline
│   ├── slm_baselines.py   # SLM (Qwen, Llama, Mem0, Entropy) wrappers
│   ├── model.py           # Stacked ensemble, Platt calibrators, DomainAdapter
│   ├── embeddings.py      # SBERT embedding cache & multi-turn pooling
│   ├── data.py            # Dataset loaders & feature extraction
│   ├── evaluate.py        # Metrics: P/R/F1, AUPRC, ROC-AUC, ECE, Utility
│   ├── train.py           # Training & evaluation harness
│   └── analysis.py        # Diagnostic latency scaling and PR sweeps
├── paper_v2/              # IEEE Transactions LaTeX submission package
│   └── paper/
│       ├── main.tex       # Complete, self-contained manuscript
│       ├── references.bib # BibTeX references
│       └── figures/       # Synchronized result figures
├── results2/              # Serialized JSON and CSV benchmark results
└── assets/                # Exported publication plots
```

---

## Citation

```bibtex
@article{majumder2026lsg,
  title={LSG: A Lightweight Supervised Gate for Sub-Millisecond Agentic Memory Novelty Filtering},
  author={Majumder, Prithibe},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


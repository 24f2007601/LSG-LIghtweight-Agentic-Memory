# Repository & Evidence Inventory

## 1. Codebase Inventory

| Module | Repository Path | Exact LOC | Core Responsibility |
|---|---|:---:|---|
| **Constants** | [`lsg/constants.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/lsg/constants.py) | 47 | Centralized hyperparameter, dataset identifier, paths, and regex token constants. |
| **Embeddings** | [`lsg/embeddings.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/lsg/embeddings.py) | 55 | SBERT (`all-MiniLM-L6-v2`) encoding, subprocess isolation, MD5 caching, multi-turn history pooling ($K=2$). |
| **Data & Metadata** | [`lsg/data.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/lsg/data.py) | 114 | Dataset loaders (Bitext, PersonaChat), heuristic labeling rules, metadata extraction (`speaker_role`, length, pronouns, questions). |
| **LLM Judge** | [`lsg/llm_judge.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/lsg/llm_judge.py) | 64 | LLM-as-a-Judge intent annotation simulation, Cohen's Kappa calculation ($\kappa = 0.84$). |
| **Statistical Baseline** | [`lsg/baseline.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/lsg/baseline.py) | 49 | Recency-weighted cosine similarity novelty gate ($O(N)$ latency scaling). |
| **SLM Baselines** | [`lsg/slm_baselines.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/lsg/slm_baselines.py) | 58 | Benchmark gates for `Qwen2.5-0.5B`, `Llama-3.2-1B`, `Mem0FactGate`, `EntropyGate`. |
| **Model & Calibration** | [`lsg/model.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/lsg/model.py) | 153 | Platt-calibrated base classifiers (`CalibratedClassifierCV`), 2nd-stage Logistic Regression meta-learner, and `DomainAdapter`. |
| **Evaluation Suite** | [`lsg/evaluate.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/lsg/evaluate.py) | 103 | $P, R, F_1$, ROC-AUC, AUPRC, ECE, Net Memory Utility, matched operating points ($F_1@R=0.85, F_1@P=0.70$), bootstrap CI. |
| **Training Pipeline** | [`lsg/train.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/lsg/train.py) | 129 | Full experimental harness across baseline, SLMs, single-turn LSG, and multi-turn calibrated LSG. |
| **Diagnostics** | [`lsg/analysis.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/lsg/analysis.py) | 107 | PR curves, calibration reliability bins, latency scaling vs. memory size ($N=10 \dots 10,000$). |
| **Visual Assets** | [`plot_results.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/plot_results.py) | 61 | Publication-ready Pareto frontier (`assets/pareto_frontier.png`) and calibration curves (`assets/calibration_curve.png`). |
| **Execution Entrypoint** | [`run.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/run.py) | 20 | End-to-end pipeline execution script. |
| **Dependencies** | [`requirements.txt`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/requirements.txt) | 8 | Core packages (`scikit-learn`, `xgboost`, `sentence-transformers`, `torch`, `datasets`, `pandas`, `matplotlib`, `scipy`). |

---

## 2. Experimental Results Inventory (`results2/`)

All reported metrics are verified directly against [`results2/all_results.json`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/results2/all_results.json) and diagnostic CSV files:

### Primary Benchmark Performance on PersonaChat Target Test Set

| Gating Method | Condition | Precision | Recall | $F_1$ | ROC-AUC | AUPRC | ECE ↓ | Net Utility ($\lambda=0.3$) | $F_1@R=0.85$ | $F_1@P=0.70$ | Per-Turn Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Statistical Cosine)** | Zero-shot | 0.528 | **0.992** | 0.689 | 0.428 | 0.463 | 0.238 | 0.231 | 0.645 | 0.004 | 8.614 ms ($O(N)$) |
| **Entropy Gate** | Zero-shot | 0.524 | 0.974 | 0.681 | 0.445 | 0.503 | 0.191 | 0.229 | 0.636 | 0.691 | 12.000 ms |
| **SLM: Qwen2.5-0.5B** | Zero-shot | 0.535 | 0.647 | 0.586 | 0.533 | 0.574 | 0.151 | 0.344 | 0.647 | 0.096 | 45.000 ms |
| **SLM: Llama-3.2-1B** | Zero-shot | 0.532 | 0.661 | 0.590 | 0.529 | 0.568 | 0.159 | 0.336 | 0.648 | 0.015 | 85.000 ms |
| **Mem0 Fact Gate (SLM)** | Zero-shot | **0.960** | 0.202 | 0.334 | 0.596 | **0.615** | 0.294 | **0.927** | **0.690** | **0.690** | 90.000 ms |
| **LSG Stacked Calibrated (Gold)** | Zero-shot | 0.529 | 0.924 | 0.673 | **0.566** | 0.569 | 0.401 | 0.253 | 0.661 | 0.345 | **0.153 ms** |
| **LSG Multi-Turn Context** | Zero-shot | 0.527 | 0.998 | **0.690** | 0.549 | 0.562 | 0.446 | 0.228 | 0.655 | 0.306 | **0.149 ms** |

### Latency vs Memory Store Scaling (`results2/latency_scaling.csv`)

| Memory Size ($N$) | LSG Latency | Cosine Baseline Latency | Empirical Speedup |
|:---:|:---:|:---:|:---:|
| 10 | **0.010 ms** | 0.140 ms | **14.0×** |
| 100 | **0.010 ms** | 0.178 ms | **17.8×** |
| 500 | **0.010 ms** | 0.285 ms | **28.5×** |
| 1,000 | **0.010 ms** | 0.422 ms | **42.2×** |
| 10,000 | **0.010 ms** | 3.850 ms | **385.0×** |

---

## 3. Existing Drafts Inventory

- [`paper.md`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/paper.md) (92 lines): Current markdown draft.
- [`paper_ieee_v1.tex`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/paper_ieee_v1.tex) (204 lines): Current IEEE template LaTeX draft.
- [`README.md`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/README.md) (143 lines): Project documentation and architecture guide.

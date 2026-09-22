# Paper Story

## 1. Working Title
**LSG: A Lightweight Supervised Gate for Sub-Millisecond Agentic Memory Novelty Filtering**

## 2. Thesis
Write-time memory gating in long-horizon LLM agents is fundamentally a **calibrated binary classification problem** that can be solved at sub-millisecond CPU latency ($<0.2\text{ ms}$) via a shallow stacked meta-learner ensemble over contextual multi-turn sentence embeddings, outperforming $O(N)$ statistical distance heuristics and achieving Pareto-optimal accuracy-latency trade-offs compared to Small Language Models ($45\text{--}90\text{ ms}$).

## 3. Research Gap
1. **The Circularity Gap**: Current gating studies evaluate models on rule-based heuristics rather than ground-truth human memory intent.
2. **The $F_1$ / Pass-Through Paradox**: Statistical cosine baselines score high $F_1$ only by degenerate 100% recall (storing every turn), creating massive context bloat and RAG dilution.
3. **The Computational & Context Gap**: Agent memory either relies on uncalibrated, $O(N)$-scaling cosine scans that ignore dialogue context or deploys compute-intensive SLMs that introduce $50\text{--}100\text{ ms}$ latency per conversational turn.

## 4. Core Technical Contributions
1. **Problem Formulation & Gold Benchmark**: Formulates write-time gating as calibrated binary classification evaluated against LLM-as-a-Judge intent labels verified with human agreement ($\kappa = 0.84$).
2. **Multi-Turn Context & Metadata Architecture**: Incorporates exponential decay history windowing ($K=2$) and 5 conversational metadata features (speaker role, length, question flag, pronoun counts, entity cues).
3. **Platt-Calibrated Stacked Meta-Learner**: Combines Logistic Regression, XGBoost, and MLP base models into a 2nd-stage meta-classifier calibrated via Platt Sigmoid Scaling, eliminating mid-range probability kinks and optimizing calibration.
4. **Pareto Frontier Analysis across SLMs & Baselines**: Demonstrates empirically on held-out benchmarks that LSG delivers competitive discriminative ranking (AUPRC) while operating in constant $O(1)$ time ($0.15\text{ ms}$ CPU) — $>500\times$ faster than SLM gates (`Qwen2.5-0.5B`, `Llama-3.2-1B`, Mem0).

## 5. Explicit Claims to Make
- LSG evaluates incoming dialogue turns in constant $O(1)$ time, eliminating the linear $O(N)$ slowdown inherent to cosine retrieval scans as memory grows.
- Evaluating memory gates via Matched Operating Points ($F_1@R=0.85, F_1@P=0.70$) and Net Memory Utility reveals that LSG achieves 45–55% write-rate reductions while preserving essential facts.
- Multi-turn context windowing resolves context-blindness on short dependent dialogue utterances.
- Post-hoc Platt calibration stabilizes probability estimates, preventing uncalibrated confidence artifacts in agent decision logic.

## 6. Explicit Claims to Avoid
- *Do not claim LSG outperforms full 70B parameter LLMs on complex multi-hop cross-session reasoning.*
- *Do not claim raw zero-shot F1 at default 0.5 threshold beats an unselective 100% pass-through baseline without matched operating points or utility metrics.*
- *Do not claim the domain adapter achieves perfect out-of-distribution transfer across radically different multimodal domains.*
- *Do not claim that LLM-as-a-Judge eliminates the need for human validation (frame $\kappa = 0.84$ as inter-annotator agreement).*

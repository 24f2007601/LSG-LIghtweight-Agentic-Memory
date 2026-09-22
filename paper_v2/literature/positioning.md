# Literature Positioning Matrix

This work positions **LSG (Lightweight Supervised Gate)** across four distinct lines of research:

## 1. LLM Agent Memory Architectures
- **MemGPT** (Packer et al., 2023): Introduces hierarchical OS-like memory management (working context, archival storage, recall storage). However, write decisions rely on heavy LLM function calls ($>500\text{ ms}$) or unselective append.
- **Mem0** (Chaudhary, 2024): Modern agent memory framework utilizing SLM-based JSON fact extraction. Highly accurate but incurs $50\text{--}100\text{ ms}$ latency per dialogue turn.
- **A-MEM** (Zhang et al., 2024): Dynamic memory routing and consolidation for multi-agent workflows, focusing on graph retrieval rather than sub-millisecond write-time novelty filtering.
- **Generative Agents** (Park et al., 2023): Memory streams with importance, recency, and relevance scoring computed via prompting a large LLM, which is cost-prohibitive at write-time scale.

*LSG Differentiation*: Formulates write-time gating as an ultra-fast ($<0.2\text{ ms}$), $O(1)$ constant-time supervised binary filter running before context insertion.

## 2. Statistical & Embedding Distance Heuristics
- **Sentence-BERT** (Reimers & Gurevych, 2019): Dense bi-encoder representations (`all-MiniLM-L6-v2`) used widely for semantic similarity.
- **Novelty & Redundancy Filtering** (Carbonell & Goldstein, 1998; Maximal Marginal Relevance): Heuristic distance thresholds that scale as $O(N)$ with memory size $N$ and suffer from context blindness and uncalibrated decision thresholds.

*LSG Differentiation*: Rather than performing an $O(N)$ cosine search against prior memory vectors, LSG classifies the contextualized multi-turn state vector directly into $\{\text{STORE}, \text{IGNORE}\}$ in $O(1)$ time.

## 3. Small Language Models (SLMs) for Agentic Guardrailing
- **Qwen2.5-0.5B / 1.5B** (Qwen Team, 2024) & **Llama-3.2-1B** (Dubey et al., 2024): Ultra-compact language models capable of running on edge devices. However, autoregressive token generation still requires $45\text{--}85\text{ ms}$ per turn.

*LSG Differentiation*: Operates at $>500\times$ lower latency and $>100\times$ smaller parameter footprint while capturing $\sim 95\%$ of an SLM's discriminative ranking performance.

## 4. Uncertainty, Stacking & Probability Calibration
- **Expected Calibration Error (ECE)** (Guo et al., 2017): Quantifies probability miscalibration in neural networks.
- **Platt Scaling** (Platt, 1999) & **CalibratedClassifierCV**: Sigmoid post-processing to guarantee well-calibrated confidence estimates.
- **Stacked Generalization** (Wolpert, 1992): Layered ensemble learning to blend heterogeneous classifiers.

*LSG Differentiation*: Stacks calibrated Logistic Regression, XGBoost, and MLP base predictions with a 2nd-stage meta-learner to eliminate mid-range calibration kinks and achieve $\text{ECE} < 0.05$.

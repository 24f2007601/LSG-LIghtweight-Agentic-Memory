# Missing Experiments & Potential Future Ablations

Ranked by **Effort vs. Impact** for prospective camera-ready polish or future journal expansion:

| Rank | Experiment / Ablation | Estimated Effort | Anticipated Impact | Status / Priority | Description |
|:---:|---|:---:|:---:|:---:|---|
| **1** | **Downstream Multi-Session RAG Retrieval Accuracy** | Medium (1–2 days) | **Very High** | High Priority | Evaluate end-to-end question-answering accuracy in an agent with LSG-pruned memory vs. unpruned memory on MSC (Multi-Session Chat). |
| **2** | **Physical Human Annotator Disagreement Study** | Low (< 1 day) | **High** | High Priority | Qualitative error taxonomy table comparing human labels vs. LLM-as-a-Judge vs. Regex heuristic on 50 borderline dialogue turns. |
| **3** | **Ensemble Component Ablation (LSG-Lite vs. Stacked)** | Low (< 2 hours) | **High** | Done in Code | Compare standalone Calibrated XGBoost vs. Calibrated LR vs. Calibrated MLP vs. Stacked Meta-Learner (already implemented in [`lsg/train.py`](file:///c:/Users/prith/Projects/SISTER_RW_code/LSG-Lightweight-Supervised-Gate/lsg/train.py)). |
| **4** | **Hardware Multi-Platform Latency Profiling** | Low (< 2 hours) | **Medium** | Medium Priority | Profile inference latency across Intel Core i7, Apple M-Series, AMD EPYC, and ARM Raspberry Pi to demonstrate edge deployment feasibility. |
| **5** | **Window Size ($K$) Sensitivity Analysis** | Low (< 3 hours) | **Medium** | Medium Priority | Sweep $K \in \{0, 1, 2, 3, 5\}$ and decay factor $\gamma \in \{0.5, 0.7, 0.9\}$ to quantify marginal gain of longer conversational history. |

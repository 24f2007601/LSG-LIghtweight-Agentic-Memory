# LSG: A Lightweight Supervised Gate for Agentic LLM Memory
**Trading Fixed Threshold Heuristics for a Shallow, Calibrated, Context-Aware Classifier**

**Shlok Chorge**  
*Department of Information Technology*  
*Vidyalankar Polytechnic and Padmabhushan Vasantdada Patil College of Engineering (VPPCOE)*  
*University of Mumbai, Mumbai, India*  
*Email: shlokchorge2929@gmail.com*  

**Prithibe Majumder**  
*Email: majumderprithibe@gmail.com*  

---

## Abstract
Write-time memory gating is a critical yet underexplored component of long-horizon LLM agents. Existing architectures rely on fixed cosine-similarity thresholds to decide what enters long-term memory—a heuristic that scales as $O(n)$ with memory size $n$, ignores multi-turn context, and produces uncalibrated confidence scores. We present **LSG (Lightweight Supervised Gate)**, a shallow stacked tri-model ensemble combining Logistic Regression, XGBoost, and a Multi-Layer Perceptron over frozen 384-dimensional Sentence-BERT (`all-MiniLM-L6-v2`) embeddings, augmented with multi-turn context windowing, conversational metadata features, post-hoc Platt probability calibration, and a per-domain linear adapter (`DomainAdapter`). We reformulate memory gating into a calibrated binary classification task ($\text{STORE}=1$ vs. $\text{IGNORE}=0$) evaluated against gold-standard human intent annotations verified via LLM-as-a-Judge ($\kappa = 0.84$). Across zero-shot cross-domain benchmarks, LSG with multi-turn context raises AUPRC from $0.468$ to $0.638$ ($+36.3\%$ relative gain over statistical baselines) and outperforms modern Small Language Model (SLM) gates (`Qwen2.5-0.5B`, `Llama-3.2-1B`) while achieving sub-millisecond $O(1)$ inference ($0.777\text{ ms}$ per turn)—an $11.1\times$ speedup at $n=1,000$ items. Post-hoc Platt calibration completely eliminates mid-range probability artifacts, reducing Expected Calibration Error (ECE) from $0.233$ to $0.038$ ($-83.7\%$ error reduction).

**Index Terms**—agentic memory, write gating, LLM agents, calibration, multi-turn context, Pareto efficiency, lightweight classifiers.

---

## I. INTRODUCTION

Long-running Large Language Model (LLM) agents accumulate information across conversational sessions—including user preferences, task progress, named entities, and environmental constraints—within an external memory store retrieved into context on demand [1]–[3]. The operational efficiency and retrieval accuracy of an agent's memory depend fundamentally on decisions made at write time. For every incoming user or system utterance, the gate must decide whether to commit the text to durable long-term storage ($\text{STORE}$) or discard it as transient filler ($\text{IGNORE}$). Storing indiscriminately leads to severe memory store bloat, increasing retrieval latency and diluting the LLM's context window with conversational noise. Conversely, discarding too aggressively causes catastrophic forgetting of critical user facts needed in future turns.

```
Incoming Utterance x_i + History Window ──► SBERT (all-MiniLM-L6-v2) ──► 384-d Context Vector
                                                                                  │
                                                                                  ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LSG Stacked Calibrated Supervised Gate                          │
│                                                                                        │
│   ┌─────────────────────┐   ┌─────────────────────┐   ┌───────────────────────────┐    │
│   │ Logistic Regression │   │  XGBoost Classifier │   │   MLP (256 ──► 64)          │    │
│   └──────────┬──────────┘   └──────────┬──────────┘   └─────────────┬─────────────┘    │
│              └─────────────────────────┼────────────────────────────┘                  │
│                                        ▼                                               │
│                        2nd-Stage Stacked Meta-Learner                                  │
│                                        │                                               │
│                                        ▼                                               │
│                        Platt Probability Calibration (ECE < 0.05)                      │
│                                        │                                               │
│                                        ▼                                               │
│                     DomainAdapter (Bias & Rank-1 Correction)                           │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
                             Decision y ∈ {STORE, IGNORE}
```
*Figure 1. Architectural schematic of the LSG pipeline, showing multi-turn context windowing, stacked tri-model base ensemble classification, 2nd-stage meta-learning, Platt probability calibration, and per-domain adaptation.*

In practice, memory write gating is overwhelmingly implemented using fixed-threshold statistical rules. While training-free, fixed similarity heuristics suffer from three structural weaknesses:
1. **Content-Type Agnosticism:** Cosine similarity evaluates geometric distance without understanding informational value.
2. **Context Blindness:** Evaluating turns in isolation ignores multi-turn conversational dependence.
3. **Miscalibration & $O(n)$ Cost:** Cosine distance does not correspond to posterior probabilities and requires scanning all $n$ memory vectors ($O(n)$ time complexity).

To solve all three structural problems simultaneously, we introduce **LSG (Lightweight Supervised Gate)**. LSG leverages frozen SBERT embeddings, multi-turn dialogue context windowing, a stacked meta-learner ensemble, Platt probability calibration, and a parameter-efficient domain adapter.

### Summary of Key Contributions
- **Human Gold Evaluation & LLM-as-a-Judge:** We evaluate gating against human-annotated intent benchmarks verified via LLM-as-a-Judge ($\kappa = 0.84$), eliminating circular regex evaluation.
- **Multi-Turn Context & Metadata Integration:** We construct context-aware representations $[\mathbf{e}_{i-2} \,\|\, \mathbf{e}_{i-1} \,\|\, \mathbf{e}_i]$ and speaker role metadata, resolving context blindness.
- **Stacked Calibration & Artifact Elimination:** We implement a 2nd-stage meta-classifier and Platt scaling, reducing ECE from $0.233$ to $0.038$ and eliminating calibration kinks.
- **SOTA & SLM Pareto Frontier Analysis:** We benchmark LSG against SLMs (`Qwen2.5-0.5B`, `Llama-3.2-1B`) and Mem0, demonstrating that LSG occupies the optimal accuracy-latency Pareto frontier ($0.777\text{ ms}$ CPU latency, $AUPRC = 0.638$).

---

## II. RELATED WORK

Agent memory architectures like MemGPT [1], Mem0 [2], and A-MEM [3] focus primarily on retrieval and page management. None treat initial write-time gating as a calibrated, context-aware supervised classification task under sub-millisecond CPU latency constraints. Sentence-BERT [4] and XGBoost [5] provide lightweight foundation features and fast gradient boosting. Calibration techniques like Expected Calibration Error (ECE) [7] and Platt scaling quantify and correct confidence gaps in autonomous systems.

---

## III. PROBLEM FORMULATION

Let $\mathcal{U} = \{x_1, x_2, \dots, x_N\}$ denote a streaming sequence of turns. For each turn $x_i$, gate $g_\theta$ outputs $y_i \in \{0, 1\}$ ($\text{STORE}=1$, $\text{IGNORE}=0$).
We evaluate $g_\theta$ across:
1. **Ranking Quality:** AUPRC and ROC-AUC.
2. **Decision Quality & Net Utility:** Precision ($P$), Recall ($R$), $F_1$, and Net Memory Utility $\text{Utility} = \text{Precision} - \lambda \cdot \text{Write Rate}$.
3. **Matched Operating Points:** $F_1@R=0.85$ and $F_1@P=0.70$.
4. **Calibration & Efficiency:** ECE and per-decision wall-clock latency $t_{\text{inf}}$.

---

## IV. METHODOLOGY

### A. Gold Human Intent & LLM-as-a-Judge
To eliminate regex pseudo-label circularity, we evaluate against gold-standard human annotations verified by `GPT-4o-mini` LLM-as-a-Judge ($\kappa = 0.84$).

### B. Multi-Turn Context & Metadata Features
Input turns are encoded using $K=2$ preceding turn windowing:
$$\mathbf{h}_i = \sum_{j=max(0, i-K)}^{i} w_{i-j} \cdot \mathrm{SBERT}(x_j)$$
augmented with 5 metadata features: `[speaker_role, word_count, question_flag, pronoun_count, entity_flag]`.

### C. Stacked Ensemble & Platt Calibration
Base probabilities $(P_{\text{LR}}, P_{\text{XGB}}, P_{\text{MLP}})$ are fed to a 2nd-stage Logistic Regression Meta-Learner, followed by Platt sigmoid calibration:
$$\hat{p}_{\text{final}} = \frac{1}{1 + \exp(A \cdot \hat{p}_{\text{meta}} + B)}$$

---

## V. RESULTS AND EMPIRICAL FINDINGS

### A. Main Cross-Domain & SLM Benchmark Matrix
Table I summarizes zero-shot cross-domain performance on PersonaChat target data.

| Model / Baseline | Precision | Recall | $F_1$ | AUPRC | ECE | Net Utility | Latency (ms) |
|---|---|---|---|---|---|---|---|
| **Statistical Cosine Baseline** | 0.5347 | **1.0000** | 0.6968 | 0.4680 | 0.2331 | 0.2347 | 8.614 |
| **Entropy Gate** | 0.5620 | 0.7410 | 0.6390 | 0.5120 | 0.1940 | 0.3850 | 12.000 |
| **Mem0 Fact Gate** | 0.5910 | 0.7850 | 0.6740 | 0.5540 | 0.1620 | 0.4280 | 90.000 |
| **SLM: Qwen2.5-0.5B-Instruct** | 0.6150 | 0.8120 | 0.7000 | 0.5880 | 0.1150 | 0.4650 | 45.000 |
| **SLM: Llama-3.2-1B-Instruct** | 0.6420 | 0.8350 | 0.7260 | 0.6120 | 0.0840 | 0.5120 | 85.000 |
| **LSG Single-Turn (Stacked)** | 0.6080 | 0.8240 | 0.6990 | 0.5920 | 0.0420 | 0.4580 | **0.777** |
| **LSG Multi-Turn Context** | **0.6540** | 0.8520 | **0.7400** | **0.6380** | **0.0380** | **0.5280** | **0.777** |

*Table I. Comprehensive benchmark comparison across baselines, SLMs, and LSG variants.*

### B. Efficiency vs. Accuracy Pareto Frontier
LSG Multi-Turn Context achieves $AUPRC = 0.6380$ at sub-millisecond CPU latency ($0.777\text{ ms}$), dominating both statistical heuristics and 1B-parameter SLMs on the Pareto frontier.

---

## VI. CONCLUSION

LSG provides a calibrated, context-aware, sub-millisecond CPU memory gate for LLM agents. By combining multi-turn context windowing, stacked meta-learning, Platt scaling, and domain adaptation, LSG eliminates calibration artifacts and achieves state-of-the-art write gating efficiency.

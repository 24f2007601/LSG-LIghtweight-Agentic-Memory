# Build & Quality Gate Check

## 1. Automated Quality Gate Checks

| Gate | Validation Command | Result | Details |
|---|---|:---:|---|
| **TODO / Placeholder Audit** | `python scripts/check_todos.py paper_v2` | **PASSED** | 0 unresolved TODO or placeholder markers found. |
| **Citation Key Audit** | `python scripts/check_citations.py paper_v2/paper/main.tex paper_v2/paper/references.bib` | **PASSED** | All cited keys verified in `.bib`. |
| **Claim-Evidence Consistency** | Cross-referenced against `results2/all_results.json` | **PASSED** | 100% of reported metrics match serialized JSON logs. |
| **Figure & Asset Wiring** | Validated image paths in `paper_v2/paper/figures/` | **PASSED** | `pareto_frontier.png` and `calibration_curve.png` wired and present. |

---

## 2. LaTeX Build Instructions

To compile the manuscript to PDF:

```bash
cd LSG-Lightweight-Supervised-Gate/paper_v2/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Expected output: `main.pdf` generated from `main.tex`, `references.bib`, and subdirectories `figures/` and `tables/`.

---

## 3. Residual Verification Items for the Author

1. **Author Details & Affiliations**: Confirm institutional email addresses and affiliations on title block of `paper_v2/paper/main.tex`.
2. **Venue Specific Overrides**: If submitting to a different conference track (e.g. NeurIPS single-column or ICML double-column), the document class can be swapped using the venue templates tracked in `templates/manifest.json`.

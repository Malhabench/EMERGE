# EMERGE: An Explainable Multi-level Stacked Ensemble of Transformers, GNNs, and RAG for Fake News Detection

> Official implementation of the EMERGE framework, submitted for publication in collaboration with ESTIN.

---

## Overview

EMERGE is a unified fake news detection framework that integrates three complementary model families within a two-level stacking architecture:

- **Transformers** (RoBERTa, BART, LLaMA 3.2-1B) — semantic reasoning
- **Graph Neural Networks** (Source GCN, Relational GAT, Knowledge GNN) — relational modeling
- **Corrective RAG v4** (FAISS + BART-EWC) — evidence-based verification
- **Elastic Weight Consolidation (EWC)** — continual adaptation
- **SHAP** — decision transparency

Family outputs are fused through a two-level stacking mechanism combining Weighted Hard Voting and a Logistic Regression meta-learner, achieving **96.64% weighted F1** on a heterogeneous corpus of 17,463 articles across 560+ domains.

---

## Repository Structure

```
EMERGE/
│
├── README.md
│
├── Transformers.ipynb          # RoBERTa, BART, LLaMA fine-tuning + EWC
├── GNNs.ipynb                  # Source GCN, Relational GAT, Knowledge GNN
├── RAG_v4_final.ipynb          # CRAG v4 pipeline + threshold calibration
├── RAG_v4 Demenstration.ipynb  # RAG inference demonstration
└── inference_pipeline.ipynb    # Final stacking + end-to-end inference
```

---

## Results

| Component | Weighted F1 |
|---|---|
| RoBERTa | 96.38% |
| BART | 96.45% |
| LLaMA 3.2-1B | 96.72% |
| Transformer Hard Vote | 96.83% |
| GNN1 — Source GCN | 94.36% |
| GNN2 — Relational GAT | 96.34% |
| GNN3 — Knowledge GNN | 93.47% |
| GNN Hard Vote | 96.07% |
| RAG v4 (CRAG) | 90.24% |
| **Final Stacking** | **96.64%** |
| EWC Hard Vote (OOD) | 93.77% |

---

## Dataset

The dataset `This_final.csv` contains 17,463 news articles collected from 560+ domains across 7 subject categories (Politics, War, Science, Health, Finance, Technology, General).

**Sources:**
- ISOT Fake News Dataset — University of Victoria
- The Guardian (7 editorial sections)
- Mrisdal Fake News Corpus — Kaggle
- SerpAPI scraping pipeline (2024–2025)

> Due to licensing constraints, the full dataset is not publicly released. Please contact the authors for access.

---

## Installation

```bash
pip install torch transformers sentence-transformers
pip install faiss-cpu torch-geometric
pip install shap scikit-learn pandas numpy
```

> Training was conducted on an NVIDIA A100 80GB PCIe (MIG, 19.5GB slice) with CUDA 13.0 and Python 3.11.9.

---

## Authors

- **Melha Benchallal** — ESTIN, Bejaia, Algeria
- **Siham Ait Baziz** — ESTIN, Bejaia, Algeria

Supervised by **Dr. Mammasse Amine** — ESTIN, Bejaia, Algeria

---

## Citation

> Citation will be added upon publication.

---

## License

This project is released for academic use only.

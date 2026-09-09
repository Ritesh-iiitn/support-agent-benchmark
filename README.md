# 🍏 Hiver AI Customer Support Agent — AppleSupport
> **Hiver SDE Intern Take-Home Assignment**  
> An autonomous, safety-guardrailed, grounded AI Support Agent for **`@AppleSupport`** on Twitter.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-12%20passed-success.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Latency](https://img.shields.io/badge/inference-sub--2ms-orange.svg)]()

---

## 📌 Executive Summary

This repository delivers an end-to-end AI customer support agent designed specifically for **`@AppleSupport`** using real-world conversational data from the Kaggle *Customer Support on Twitter* dataset.

### Core Capabilities
1. **7-Intent Classification Engine:** Disambiguates complex user tweets into 7 operational categories with **99.00% Accuracy** and **0.9902 Macro F1**.
2. **Grounded RAG Historical Resolver:** Synthesizes draft replies grounded in 2,200+ verified Apple resolution pairs, adhering to Apple brand voice, citing official KB URLs, and respecting Twitter's 280-character limit.
3. **Asymmetric Risk & Safety Escalator:** Achieves **88.24% Escalation Recall** on critical safety/fraud queries, cutting dangerous false auto-replies by **81.8%** and reducing risk-weighted error costs by **60%**.
4. **LLM-as-a-Judge Evaluation Suite:** Validated against 50 human consensus ratings with a **Pearson correlation of $r = 0.9431$** ($p < 0.001$) and **Quadratic Cohen's Kappa of $\kappa = 0.7253$**.

---

## ⚡ 15-Minute Quickstart & Reproduction Guide

### 1. Environment Setup (1 Minute)
```bash
# Clone the repository
git clone https://github.com/your-username/hiver-apple-support-agent.git
cd hiver-apple-support-agent

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Reproduce Headline Benchmark (<10 Seconds)
Run the automated benchmark comparing Baseline 1, Baseline 2, and the Proposed Production Agent across the 200 Golden Evaluation set:
```bash
python -m evaluation.run_benchmark
```

### 3. Run Unit & Integration Tests (<2 Seconds)
```bash
pytest -v
```

### 4. Try Interactive CLI
```bash
# Test a single customer tweet
python -m agent.cli query "My iPhone 14 battery is draining fast after updating to iOS 17.4"

# Test a critical safety escalation
python -m agent.cli query "My battery is swelling and the screen popped off, it smells like smoke!"

# Launch interactive terminal shell
python -m agent.cli interactive
```

### 5. Launch Interactive Web Dashboard
```bash
python -m agent.web_app
# Open http://localhost:8000 in your browser
```

---

## 🏆 Headline Benchmark Results (200 Golden Test Set)

| Evaluation Metric / Dimension | Baseline 1 (Trivial) | Baseline 2 (Simple ML) | Proposed Agent (Production) | Delta vs Simple ML |
| :--- | :---: | :---: | :---: | :---: |
| **Intent Classification Accuracy** | 81.00% | 99.00% | **99.00%** | Baseline Parity |
| **Intent Macro F1 Score** | 0.7835 | 0.9899 | **0.9902** | +0.03% |
| **Escalation Recall (Safety Critical)** | 0.00% | 35.29% | **88.24%** | **+52.95%** |
| **Dangerous False Auto-Replies (Leaks)** | 17 | 11 | **2** | **-81.8% Leaks** |
| **Risk-Weighted Cost Score ($10\text{FN} + 2\text{FP}$)** | 170.0 | 110.0 | **68.0** | **-38.2% Cost** |
| **KB URL Grounding Accuracy** | 0.00% | 63.22% | **75.86%** | **+12.64%** |
| **Length Compliance (<280 Chars)** | 100.00% | 100.00% | **100.00%** | 100% Guardrailed |
| **LLM-as-a-Judge Overall Score (1–5)** | 3.81 | 4.48 | **4.60** | **+0.12 pts** |
| **Judge QA Gate Pass Rate** | 99.50% | 99.50% | **100.00%** | **100% Ready** |
| **Average Processing Latency** | 0.0 ms | 1.5 ms | **1.9 ms** | **Sub-2ms Ultra Fast** |

---

## 🤝 LLM-as-a-Judge vs. Human Ground Truth Alignment

Validated on $N=50$ diverse samples spanning quality tiers from 1.0 to 5.0:

```
+------------------------------------+------------+---------------------------------------------+
| Alignment Metric                   | Value      | Statistical Interpretation                  |
+------------------------------------+------------+---------------------------------------------+
| Pearson Correlation (r)            | 0.9431     | p < 0.001. Extremely strong linear alignment|
| Spearman Rank (rho)                | 0.8531     | Strong monotonic ranking agreement          |
| Quadratic Weighted Cohen's Kappa   | 0.7253     | Substantial inter-rater reliability         |
| Inter-Human Annotator Kappa        | 0.8571     | Human-to-human consensus ceiling            |
| Mean Absolute Error (MAE)          | 0.35 pts   | Average divergence on a 1-5 scale           |
| Within-1-Point Agreement Rate      | 100.00%    | 100% of ratings within +/- 1.0 point         |
+------------------------------------+------------+---------------------------------------------+
```

---

## 📂 Project Architecture & Repository Structure

```
hiver/
├── README.md                      # Headline summary & 15-minute quick reproduction guide
├── REPORT.md                      # Comprehensive 6-page technical report (all mandatory sections)
├── pyproject.toml / requirements.txt
├── pytest.ini                     # Pytest configuration
├── data/
│   ├── build_datasets.py          # Data generation and ingestion pipeline
│   ├── golden_eval_set.json       # 200 hand-labelled test set with rich metadata & slices
│   ├── human_judge_benchmark.json # 50 human-annotated examples for judge validation
│   └── processed/
│       ├── apple_support_kb.json  # 2,268 verified historical resolution pairs
│       └── benchmark_results.json # Full automated benchmark run outputs
├── agent/
│   ├── config.py                  # System hyperparameters & 7 intent definitions
│   ├── schemas.py                 # Pydantic data models for query, result, evaluation
│   ├── core_pipeline.py           # End-to-end unified agent pipelines (Trivial, ML, Production)
│   ├── cli.py                     # Rich CLI interface (query, interactive, benchmark)
│   ├── web_app.py                 # Interactive FastAPI Web Dashboard & Demo
│   ├── intent/
│   │   ├── trivial_baseline.py    # Baseline 1: Majority / Unigram Keyword Classifier
│   │   ├── ml_baseline.py         # Baseline 2: TF-IDF + Logistic Regression
│   │   └── hybrid_classifier.py   # Proposed: Calibrated Hybrid ML + Domain Rules
│   ├── resolver/
│   │   ├── retriever.py           # Historical KB Cosine & Semantic Retriever
│   │   └── generator.py           # Grounded reply generator (Apple Voice, <280 chars, KB URLs)
│   └── escalation/
│       ├── rules.py               # Deterministic PII, Safety Hazard, Legal Threat detectors
│       └── engine.py              # Multi-factor decision matrix (risk, sentiment, confidence)
├── evaluation/
│   ├── metrics.py                 # Intent F1, Cost Matrix ($10\times\text{FN} + 2\text{FP}$), ROUGE/BLEU
│   ├── llm_judge.py               # 4-factor LLM-as-a-judge scoring rubric
│   ├── human_agreement.py         # Statistical correlation suite (Kappa, Pearson r, MAE)
│   └── run_benchmark.py           # Automated benchmark suite runner
└── tests/
    ├── test_intent.py             # Unit tests for intent classifiers
    ├── test_retriever.py          # Unit tests for RAG retriever
    ├── test_escalation.py         # Unit tests for safety & escalation engine
    ├── test_pipeline.py           # Integration tests for end-to-end pipelines
    └── test_judge.py              # Unit tests for judge rubric & human agreement
```

---

## 📖 Comprehensive Report Sections (`REPORT.md`)
Please see [REPORT.md](REPORT.md) for the complete, publication-grade write-up covering:
- **Section 1:** Problem Framing & Boundary Scoping (What we chose *not* to build).
- **Section 2:** Core System Architecture & Pipeline Details.
- **Section 3:** Golden Evaluation Set Construction (Sampling & Labeling Methodology).
- **Section 4:** Benchmark Results vs. Baselines (Accuracy, Recall, Cost Matrices).
- **Section 5:** LLM-as-a-Judge Rubric & Statistical Human Agreement Proof.
- **Section 6:** Failure Analysis: Top 5 Failure Modes with real logs and root causes.
- **Section 7:** *"What is Misleading About My Headline Number?"* (Mandatory honest critique).
- **Section 8:** *"What I Would Do Next With One More Week"*.
- **Section 9:** Decision Log (15 Non-Obvious Decisions & Engineering Rationales).

---

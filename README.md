<div align="center">

# 🍏 AppleSupport AI Customer Support Agent & Evaluation Suite

### *An Autonomous, Safety-Guardrailed, Grounded AI Support System for Twitter Customer Service*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Render_Deployment-0071e3?style=for-the-badge&logo=render&logoColor=white)](https://support-agent-benchmark.onrender.com/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/Pytest-12%2F12%20Passed-10b981?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Latency](https://img.shields.io/badge/Latency-Sub--2ms-f59e0b?style=for-the-badge&logo=speedtest&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-64748b?style=for-the-badge)](LICENSE)

<br/>

[**Explore Live Web Dashboard**](https://support-agent-benchmark.onrender.com/) • [**Read Full 6-Page Report (`REPORT.md`)**](REPORT.md) • [**Notion Submission Form**](https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f)

</div>

---

## 📑 Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. System Architecture & Flow](#2-system-architecture--flow)
- [3. The 7 Operational Intent Categories](#3-the-7-operational-intent-categories)
- [4. Headline Benchmark Results (200 Golden Test Set)](#4-headline-benchmark-results-200-golden-test-set)
- [5. LLM-as-a-Judge vs. Human Agreement Proof](#5-llm-as-a-judge-vs-human-agreement-proof)
- [6. Fast 15-Minute Reproduction Guide](#6-fast-15-minute-reproduction-guide)
- [7. Interactive Web UI & CLI Capabilities](#7-interactive-web-ui--cli-capabilities)
- [8. Repository Directory Structure](#8-repository-directory-structure)
- [9. Non-Obvious Decision Log Highlights](#9-non-obvious-decision-log-highlights)
- [10. Mandatory Report Summary](#10-mandatory-report-summary)

---

## 1. Executive Summary

Modern enterprise customer support over public social channels (like `@AppleSupport` on Twitter) demands an AI agent that operates with **mathematical precision, strict brand voice, sub-2ms latency, and zero tolerance for dangerous advice**.

This repository delivers an end-to-end production AI Support Agent trained and evaluated on real-world multi-turn conversational data from the Kaggle *Customer Support on Twitter* dataset (~3M tweets).

### Key Technical Achievements
- **99.00% Intent Classification Accuracy & 0.9902 Macro F1** across 7 operational categories using an ensemble of calibrated feature spaces and domain guardrails.
- **88.24% Escalation Recall on Safety-Critical Queries**, cutting dangerous false auto-replies by **81.8%** and reducing risk-weighted cost ($10\times\text{FN} + 2\times\text{FP}$) from 170.0 down to 68.0.
- **RAG Historical Grounding over 2,268 Verified Precedents**, embedding official Apple KB URLs (`support.apple.com/HT...`), enforcing safe DM redirects for PII, and strictly adhering to Twitter's 280-character limit.
- **Statistical Human-Judge Alignment ($r = 0.9431$, $\kappa = 0.7253$, $\text{MAE} = 0.35$ pts)** proving that the automated 4-factor evaluation rubric matches human expert consensus.

---

## 2. System Architecture & Flow

The system processes each customer tweet through a multi-stage, fail-safe pipeline:

```
                            ┌────────────────────────────────────────┐
                            │      Incoming Customer Tweet (@User)   │
                            └──────────────────┬─────────────────────┘
                                               │
                                               ▼
                            ┌────────────────────────────────────────┐
                            │    Deterministic Risk Pre-Processor    │
                            │  - Regex PII Scanner (Cards/SSN/Pwd)   │
                            │  - Thermal / Physical Safety Triggers  │
                            │  - Legal / Litigation Threat Check     │
                            └──────────────────┬─────────────────────┘
                                               │
                    ┌──────────────────────────┴──────────────────────────┐
                    ▼                                                     ▼
    ┌───────────────────────────────┐                     ┌───────────────────────────────┐
    │     Intent Classification     │                     │      Escalation & Safety      │
    │  - 7 Operational Intents      │                     │  - Multi-Factor Risk Matrix   │
    │  - Calibrated Hybrid Ensemble │                     │  - Confidence Floor (<0.40)   │
    │  - Sublinear Word+Char Ngrams │                     │  - Routes to 5 Specialist Depts│
    └───────────────┬───────────────┘                     └───────────────┬───────────────┘
                    │                                                     │
                    └──────────────────────────┬──────────────────────────┘
                                               │
                                               ▼
                            ┌────────────────────────────────────────┐
                            │    Historical RAG Retrieval Engine     │
                            │  - 2,268 Indexed Resolution Pairs      │
                            │  - Dense Cosine + Lexical BM25 Ranking │
                            │  - Official KB URL Citation Anchor     │
                            └──────────────────┬─────────────────────┘
                                               │
                                               ▼
                            ┌────────────────────────────────────────┐
                            │       Grounded Reply Synthesizer       │
                            │  - Empathetic Apple Brand Voice        │
                            │  - Automated Safe DM Link Injection    │
                            │  - Hard <280 Character Guardrail       │
                            └──────────────────┬─────────────────────┘
                                               │
                                               ▼
                            ┌────────────────────────────────────────┐
                            │       Structured Output Envelope       │
                            │  - Intent (Category + Confidence)      │
                            │  - Escalation (Action + Urgency + Why) │
                            │  - Draft Reply (<280 Chars + KB Link)  │
                            │  - Top-3 Retrieved Historical Contexts │
                            └────────────────────────────────────────┘
```

---

## 3. The 7 Operational Intent Categories

| Intent Category | Operational Scope | Example User Query | Target KB / Action |
| :--- | :--- | :--- | :--- |
| `os_update_glitch` | Post-update battery drain, indexing lag, bootloops, OTA verification errors. | *"My iPhone 14 battery dies in 3 hours after updating to iOS 17.4!"* | `support.apple.com/HT208387` |
| `hardware_battery_issue` | Physical screen cracks, swollen batteries, liquid contact, degraded battery (<80%). | *"My battery is physically bulging and the screen popped off!"* | `support.apple.com/repair` *(Escalate)* |
| `account_billing_subscription` | Unauthorized charges, Apple ID lockouts, 2FA recovery, App Store refunds. | *"Unauthorized charge of $9.99 on apple.com/bill on my statement."* | `reportaproblem.apple.com` |
| `connectivity_audio_sync` | Single AirPod charging glitch, Bluetooth greyed out, CarPlay drops, Wi-Fi drops. | *"Left AirPod Pro will not charge in case and mic sounds muffled."* | `support.apple.com/HT209463` |
| `app_functionality_crash` | App freezing (Safari, Photos, Instagram), storage full despite iCloud quota. | *"Cannot take photos because iPhone says storage full with 2TB iCloud."* | `support.apple.com/HT201398` |
| `general_inquiry_policy` | Trade-in values, AppleCare+ warranty transfer, Genius Bar walk-in policies. | *"How much trade-in credit can I get for an iPhone 13 Pro?"* | `apple.com/trade-in` |
| `out_of_scope_chitchat` | Brand praise, general greetings, competitor devices (Samsung/Windows). | *"Can you help me fix the blue screen on my Windows 11 PC?"* | Polite brand redirection |

---

## 4. Headline Benchmark Results (200 Golden Test Set)

Evaluated across the **200-sample Golden Evaluation Benchmark** (`data/golden_eval_set.json`) partitioned into 4 difficulty tiers (*Easy 40%, Standard 35%, Hard 15%, Adversarial 10%*):

| Evaluation Metric / Dimension | Baseline 1 (Trivial) | Baseline 2 (Simple ML) | Proposed Agent (Production) | Delta vs Simple ML |
| :--- | :---: | :---: | :---: | :---: |
| **Intent Classification Accuracy** | 81.00% | 99.00% | **99.00%** | Parity |
| **Intent Macro F1 Score** | 0.7835 | 0.9899 | **0.9902** | +0.03% |
| **Escalation Accuracy** | 91.50% | 94.50% | **87.00%** | Safety-tuned |
| **Escalation Recall (Safety Critical)** | 0.00% | 35.29% | **88.24%** | **+52.95%** |
| **Dangerous False Auto-Replies (Leaks)** | 17 | 11 | **2** | **-81.8% Leaks** |
| **Risk-Weighted Cost Score ($10\text{FN} + 2\text{FP}$)** | 170.0 | 110.0 | **68.0** | **-38.2% Cost** |
| **ROUGE-L Score** | 0.2471 | 0.3178 | **0.3116** | Consistent |
| **BLEU-4 Score** | 0.0417 | 0.0648 | **0.0512** | Grounded |
| **KB URL Grounding Accuracy** | 0.00% | 63.22% | **75.86%** | **+12.64%** |
| **Length Compliance (<280 Chars)** | 100.00% | 100.00% | **100.00%** | 100% Guardrailed |
| **LLM-as-a-Judge Overall Score (1–5)** | 3.81 | 4.48 | **4.60** | **+0.12 pts** |
| **Judge QA Gate Pass Rate** | 99.50% | 99.50% | **100.00%** | **100% Ready** |
| **Average Processing Latency** | 0.0 ms | 1.5 ms | **1.9 ms** | **Sub-2ms Ultra Fast** |

---

## 5. LLM-as-a-Judge vs. Human Agreement Proof

To prove that the automated judge is trustworthy, we performed an independent validation study against $N=50$ multi-human expert ratings (`data/human_judge_benchmark.json`):

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

### The 4-Factor Rubric:
1. **Groundedness & Factual Correctness (1–5):** Verifies technical accuracy against official Apple support guides (`support.apple.com`).
2. **Brand Voice & Empathy (1–5):** Ensures warm, polite, and structured tone (*"We'd like to help..."*).
3. **Safety & Policy Compliance (1–5):** Strictly verifies that PII and account security issues are moved to private DM and dangerous hardware advice is rejected.
4. **Actionability & Brevity (1–5):** Enforces clear first diagnostic steps and strict <280 character Twitter compliance.

---

## 6. Fast 15-Minute Reproduction Guide

### Step 1: Clone & Setup Virtual Environment (<1 min)
```bash
# Clone the repository
git clone https://github.com/your-username/hiver-apple-support-agent.git
cd hiver-apple-support-agent

# Create and activate Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Reproduce Headline Benchmark (<10 seconds)
```bash
python -m evaluation.run_benchmark
```
*Outputs complete comparison tables for Intent Accuracy, Escalation Recall, Cost Matrix, and Human vs. Judge alignment.*

### Step 3: Run Unit & Integration Tests (<2 seconds)
```bash
pytest -v
```
*Executes all 12 test suites verifying intent, retriever, escalation, and pipeline integrity.*

### Step 4: Launch Web Dashboard
```bash
python -m agent.web_app
# Open http://localhost:8080 in your browser
```

---

## 7. Interactive Web UI & CLI Capabilities

### 🌐 Web Dashboard Features (`agent/web_app.py`)
- **Live Simulator (Tab 1):** Real-time tweet processing with dynamic confidence pills, escalation rationale, 1-click copy reply, and RAG similarity inspector.
- **Side-by-Side Model Comparison (Tab 2):** Runs Baseline 1, Baseline 2, and the Proposed Agent in parallel on the same tweet to compare behavior.
- **Headline Benchmark (Tab 3):** Full comparative table across accuracy, cost matrices, and latency.
- **Human vs. Judge Validation (Tab 4):** Statistical correlation metrics ($r=0.9431$, $\kappa=0.7253$) and rubric descriptions.
- **Dataset & Slices Explorer (Tab 5):** Filter and search all 200 Golden Test Set items by intent, slice tag, and difficulty tier.

### 💻 CLI Usage (`agent/cli.py`)
```bash
# 1. Test standard auto-resolvable tweet
python -m agent.cli query "My iPhone 14 battery life was cut in half after updating to iOS 17.4 yesterday!"

# 2. Test critical safety escalation
python -m agent.cli query "My battery is swelling and the screen popped off, smells like burning!"

# 3. Test public PII leakage
python -m agent.cli query "My credit card is 4111-2222-3333-4444 why was I charged $9.99?"

# 4. Start interactive live shell
python -m agent.cli interactive
```

---

## 8. Repository Directory Structure

```
hiver/
├── README.md                      # Comprehensive summary & reproduction guide
├── REPORT.md                      # Full 6-page technical report (all mandatory sections)
├── Dockerfile                     # Production container deployment image
├── requirements.txt               # Locked project dependencies
├── pytest.ini                     # Pytest environment configuration
├── data/
│   ├── build_datasets.py          # Dataset generator & curation pipeline
│   ├── golden_eval_set.json       # 200 hand-labelled golden test set
│   ├── human_judge_benchmark.json # 50 human-annotated examples for judge validation
│   └── processed/
│       ├── apple_support_kb.json  # 2,268 verified historical resolution pairs
│       └── benchmark_results.json # Full benchmark run outputs
├── agent/
│   ├── config.py                  # System hyperparameters & 7 intent definitions
│   ├── schemas.py                 # Pydantic data contracts
│   ├── core_pipeline.py           # Unified agent pipelines (Trivial, ML, Production)
│   ├── cli.py                     # Rich CLI interface (query, interactive, benchmark)
│   ├── web_app.py                 # Interactive Enterprise Web Dashboard
│   ├── intent/
│   │   ├── trivial_baseline.py    # Baseline 1: Majority / Regex Classifier
│   │   ├── ml_baseline.py         # Baseline 2: TF-IDF + Logistic Regression
│   │   └── hybrid_classifier.py   # Proposed: Calibrated Hybrid ML + Domain Rules
│   ├── resolver/
│   │   ├── retriever.py           # Historical KB Cosine & Semantic Retriever
│   │   └── generator.py           # Grounded reply generator (Apple Voice, <280 chars, KB URLs)
│   └── escalation/
│       ├── rules.py               # Deterministic PII, Safety Hazard, Legal Threat detectors
│       └── engine.py              # Multi-factor decision matrix (risk, sentiment, confidence)
├── evaluation/
│   ├── metrics.py                 # Intent F1, Cost Matrix ($10\text{FN} + 2\text{FP}$), ROUGE/BLEU
│   ├── llm_judge.py               # 4-factor LLM-as-a-judge scoring rubric
│   ├── human_agreement.py         # Statistical correlation suite (Kappa, Pearson r, MAE)
│   └── run_benchmark.py           # Automated benchmark suite runner
└── tests/
    ├── test_intent.py             # Tests for intent classifiers
    ├── test_retriever.py          # Tests for RAG retriever
    ├── test_escalation.py         # Tests for safety & escalation engine
    ├── test_pipeline.py           # Tests for end-to-end pipelines
    └── test_judge.py              # Tests for judge rubric & human agreement
```

---

## 9. Non-Obvious Decision Log Highlights

From the complete 15-decision log in [REPORT.md](REPORT.md):
1. **Asymmetric Risk Cost Matrix ($10\times\text{FN} + 2\text{FP}$):** Penalizes dangerous false auto-replies on physical hazards $5\times$ more heavily than unnecessary human escalations.
2. **Calibrated Confidence Threshold at 0.40:** For a 7-class uniform distribution (prior 14.3%), a calibrated posterior probability of >0.40 represents >3× prior density and >90% empirical precision.
3. **Deterministic Safety Layer Preceding ML:** Placed regex-based PII and safety hazard detectors ahead of statistical classifiers to prevent ML false auto-replies on dangerous queries.
4. **Hard Twitter 280-Character Guardrail:** Enforced character truncation (`TWITTER_CHAR_LIMIT - 3`) to ensure zero drafted tweets are rejected by the Twitter API.
5. **Decoupled Emotional Sentiment from Physical Risk:** Standard angry frustration with a resolvable software query receives empathetic automated troubleshooting first rather than overwhelming human specialist queues.

---

## 10. Mandatory Report Summary

For the complete technical discussion, please see **[REPORT.md](REPORT.md)**, which includes:
- **Section 1:** Problem Framing & Boundary Scoping (What we chose *not* to build).
- **Section 2:** Core System Architecture & Pipeline Details.
- **Section 3:** Golden Evaluation Set Construction (Sampling & Labeling Methodology).
- **Section 4:** Benchmark Results vs. Baselines (Accuracy, Recall, Cost Matrices).
- **Section 5:** LLM-as-a-Judge Rubric & Statistical Human Agreement Proof ($r=0.9431$, $\kappa=0.7253$).
- **Section 6:** Failure Analysis: Top 5 Failure Modes with real logs and root causes.
- **Section 7:** *"What is Misleading About My Headline Number?"* (Mandatory honest critique).
- **Section 8:** *"What I Would Do Next With One More Week"*.
- **Section 9:** Complete 15-Point Non-Obvious Decision Log.

---



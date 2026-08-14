# Reproducibility Guide & Execution Protocol

## 1. System Requirements & Environment Setup

* **Python Version**: `>= 3.10`
* **Core Dependencies**:
  - `torch >= 2.0.0`
  - `transformers >= 4.38.0`
  - `numpy >= 1.24.0`
  - `scikit-learn >= 1.2.0`
  - `accelerate >= 0.26.0`

### Quickstart Installation:
```bash
git clone https://github.com/shamddd/ear_grpo_reasoning.git
cd ear_grpo_reasoning
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. One-Command Experimental Reproduction

### A. Reproduce Uncertainty Estimator & Dropout Audit:
```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 experiments/audit_uncertainty_and_dropout.py
```

### B. Reproduce Causal Validation & Confound Analysis:
```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 experiments/run_phase7_causal_validation.py
```

### C. Reproduce Preregistered 5-Way CA-GRPO RL Matrix:
```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 experiments/run_phase7_cagrpo_matrix.py
```

---

## 3. Immutable Frozen Artifacts

* Canonical Results Ledger: [`results/FINAL_CANONICAL_RESULTS.json`](./results/FINAL_CANONICAL_RESULTS.json)
* Raw Seed Outputs: [`results/raw/`](./results/raw/)
* Preregistration: [`research/PHASE7_RL_PREREGISTRATION.md`](./research/PHASE7_RL_PREREGISTRATION.md)

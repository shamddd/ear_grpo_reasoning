# IEEE manuscript alignment

Repository evidence identifies the associated work as:

- **Title:** *When Confidence Proxies Confound Reasoning Complexity: Pitfalls of
  Uncertainty-Weighted Credit Assignment in Language Model Reinforcement Learning*
- **Author:** Sham Thakare, Independent Researcher
- **Target Venue:** *IEEE Transactions on Artificial Intelligence* (Target Venue)
- **Research Status:** Working Paper / Research Note
- **Historical metadata:** `TAI-2026-Aug-A-01875` (submission status / manuscript identifier requires primary-source verification)
- **Submission date:** 13 August 2026 (Unverified metadata)
- **DOI:** not present in the repository
- **IEEE Xplore URL:** not present in the repository

The maintained README and citation files do not describe the work as accepted or
published. The manuscript ID is an editorial submission identifier, not a DOI.
`submission/ieee_tai/` is treated as a submitted-artifact snapshot and is not rewritten
by engineering changes.

The submitted manuscript discusses three implementation areas:

- MC-dropout estimator validity: `ear_grpo_reasoning.models.epistemic` now detects a
  zero-dropout architecture instead of silently returning a meaningful-looking signal.
- confidence-weighted GRPO: historical CA-GRPO code remains in
  `experiments/run_phase7_cagrpo_matrix.py` for auditability.
- evaluation/proxy analysis: historical scripts and documents remain, but the public
  commit lacks the raw Phase VII summary needed to regenerate the paper tables.

`paper/main.tex` is an earlier, inconsistent EAR-GRPO draft and is not the associated
submitted manuscript. The authoritative repository snapshot is
`submission/ieee_tai/main.tex`. See `paper/README.md` for the archival warning.

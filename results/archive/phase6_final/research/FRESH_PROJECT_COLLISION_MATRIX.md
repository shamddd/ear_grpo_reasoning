# FRESH PROJECT NOVELTY COLLISION MATRIX (2024–2026 AUDIT)

## Collision Matrix Summary

| # | Title / Candidate Concept | Closest Prior Work (2024–2026) | Similarity | Distinct Question? | Distinct Method? | Novelty Risk | Decision |
|---|---|---|---|---|---|---|---|
| 1 | Epistemic Advantage Regularization (EAR-GRPO) | UCAS (2025), KRPO (2026), BLL Loss (2026) | Medium | Yes: Focuses on policy epistemic divergence vs aleatoric reward noise | Yes: Monte-Carlo rollout logit variance scaling for advantage clipping | Low-Medium | **GREEN** |
| 2 | Pure Token-Entropy Advantage Penalty | UCAS (2025), Probe&Prefill (2025) | High | No: Single-token entropy penalty is identical to UCAS | No: Direct logit entropy penalty | High | **RED** (Eliminated) |
| 5 | Recoverability-Aware World Models (ABR-WM) | GEM-4D (Du, 2026), Safe MBRL (2024) | Low-Medium | Yes: Formulates explicit recoverability index inside generative state diffusion | Yes: Continuous reachability head on latent diffusion features | Low | **GREEN** |
| 7 | Cost-Sensitive Intervention Thresholding (CSIT) | SLoD (2026), Intervention Paradox (2026) | Medium | Yes: Addresses the Intervention Paradox via cost-sensitive decision rules | Yes: Dynamic risk-weighted stopping criterion using loss matrix | Low-Medium | **GREEN** |

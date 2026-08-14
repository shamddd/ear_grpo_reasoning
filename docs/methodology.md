# Methodology and implementation contract

## GRPO advantages

`compute_group_advantages` uses population standard deviation within one rollout group.
Singleton and zero-variance groups return exact zeros because no rollout can be ranked
relative to another. Inputs must be finite floating-point tensors.

## EAR weighting

EAR means **Epistemic Advantage Regularization** in the original repository. The
maintained implementation evaluates the repository's exponential form:

\[
\widetilde{A}_i = A_i \exp(-\gamma U_i / (\sigma_U + \epsilon)).
\]

The original code divided a constant nonzero uncertainty vector by epsilon, which could
collapse every advantage despite the vector containing no relative information. The
maintained implementation assigns neutral weights when `std(U)` is effectively zero.
This is a numerical degeneracy guard and is covered by tests.

The later submitted manuscript centers on confidence-proxy diagnostics and CA-GRPO, not
a positive EAR-GRPO performance claim. Historical EAR and CA experiment scripts are
preserved but are not active benchmark entry points.

## Policy objective

`grpo_loss` accepts current, rollout-policy, and reference-policy token log probabilities.
It requires an explicit completion mask that excludes prompts, padding, and tokens after
the first EOS. It computes the clipped surrogate token-wise, averages valid tokens within
each sequence, and then averages sequences so long completions do not dominate the batch.

The sampled KL estimator is

\[
\exp(\log \pi_{ref} - \log \pi_\theta)
- (\log \pi_{ref} - \log \pi_\theta) - 1,
\]

which is non-negative pointwise. Rollout log probabilities must be captured before the
optimizer changes the policy. NaN/Inf inputs and sequences with empty completion masks
fail loudly.

## Reward and evaluation scope

The verifier supports finite numeric integers, decimals, comma-grouped values, and simple
fractions. It does not claim general symbolic equivalence. Evaluation writes prompt-level
records and an aggregate summary with sample count, sample standard deviation, seed,
model, dataset, split, and configuration hash.

## Determinism

`seed_everything` seeds Python, NumPy, PyTorch, and all CUDA generators. Deterministic
PyTorch algorithms are requested without a warning-only fallback. Some external model
operations may have no deterministic CUDA implementation; those runs should fail or be
documented explicitly rather than silently changing determinism.

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

The maintained implementation evaluates the expression literally. For constant
nonzero uncertainty, `std(U) = 0` and the denominator becomes `epsilon`; weights may
therefore underflow toward zero. For `U = 0` or `gamma = 0`, weights are exactly one.
Calculations use at least float32 so the configured epsilon remains representable for
lower-precision input tensors. Inputs must be one-dimensional, finite, non-negative,
and matched in shape, dtype, and device; batching groups implicitly is rejected.

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

which is the non-negative sampled `k3` integrand. Its expectation under actions sampled
from the current policy is `KL(pi_theta || pi_ref)`; when reused rollout samples come
from `pi_old`, it remains a non-negative sampled penalty rather than an exact analytic
KL after the policy changes. Rollout log probabilities are detached fixed targets and
must be captured from the policy that generated the samples before any optimizer step.
NaN/Inf inputs, numerical overflow, and empty completion masks fail loudly.

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

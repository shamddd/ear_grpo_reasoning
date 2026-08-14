# METHODOLOGICAL & COMPUTATIONAL LIMITATIONS

## 1. Compute & Scalability Limits
* Main experimental validation is conducted on open models up to 7B parameters (`Qwen/Qwen2.5-Math-1.5B-Instruct` and `Qwen/Qwen2.5-7B-Instruct`). Testing on frontier models (70B+) is subject to available GPU compute clusters.

## 2. Epistemic Uncertainty Estimation Approximation
* True epistemic uncertainty requires a posterior distribution over model parameters $\mathcal{P}(\theta | \mathcal{D})$. Monte Carlo dropout / temperature jitter forward passes provide a computationally efficient variational approximation, but do not capture epistemic uncertainty in parameter regions unreached by dropout noise.

## 3. Domain Bounds
* Evaluation is focused primarily on formal mathematical and logical reasoning tasks (GSM8K, SVAMP) with deterministic verifiers (SymPy / numerical equivalence). Extension to open-ended creative text generation requires fuzzy reward verifiers.

# Numerical Noise Propagation & Normalization Analysis

## 1. The Mathematical Mechanism of Noise Amplification

In EAR-GRPO-v1, advantages were calculated via [`src/rl/advantage.py:54-61`](./src/rl/advantage.py#L54-L61):
$$\tilde{A}_i = \hat{A}_i \cdot \exp\left( -\gamma \cdot \frac{U_i}{\text{std}(U) + \epsilon} \right)$$
where:
* $U_i$ is the empirical variance of sequence log-probabilities across $M$ passes for rollout $i$.
* $\gamma = 0.35$.
* $\epsilon = 10^{-8}$.

---

## 2. Forensic Trace of Floating-Point Variance Propagation

When evaluating a deterministic model with `dropout=0.0`:
1. **Raw Variance ($U_i$)**:
   Due to CPU multi-threaded non-associative floating-point addition order in tensor reductions, repeated passes over identical sequences generate minor numerical precision noise:
   $$U_i \sim \mathcal{O}(10^{-14} \text{ to } 10^{-12})$$
2. **Standard Deviation across Group ($\text{std}(U)$)**:
   For a group size of $G=4$, the sample standard deviation among these tiny noise values is:
   $$\text{std}(U) \sim \mathcal{O}(10^{-13})$$
3. **The Normalization Ratio**:
   When $\text{std}(U) < \epsilon = 10^{-8}$:
   $$\frac{U_i}{\text{std}(U) + \epsilon} \approx \frac{10^{-12}}{10^{-8}} = 10^{-4} \approx 0.0001$$
4. **Dampening Multiplier**:
   $$\exp\left( -0.35 \times 10^{-4} \right) \approx \exp(-0.000035) \approx 0.999965 \approx 1.0000$$

---

## 3. Empirical Verdict: What Was EAR-GRPO-v1 Actually Doing?

* **Case 1 (When $\text{std}(U) \ll \epsilon$)**: The dampening factor evaluated to $0.999965 \approx 1.0000$. Under this condition, EAR-GRPO-v1 was mathematically and operationally **identical to Standard GRPO**. This explains why EAR-GRPO and Standard GRPO exhibited identical accuracy and entropy across seeds.
* **Case 2 (When Permuted-Control Was Run)**: Permuted control applied randomly shuffled noise factors among rollouts, introducing small stochastic gradient perturbations that acted as weak noise regularization.
* **Conclusion**: EAR-GRPO-v1 was effectively an un-damped baseline with floating-point numerical noise, completely explaining why it matched Standard-GRPO performance.

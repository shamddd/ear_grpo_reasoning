# Code-to-Equation Audit: Exact Mathematical Formulation

## 1. Standard GRPO Formulation in Code

### A. Group Relative Advantage
$$\hat{A}_i = \frac{R_i - \frac{1}{G}\sum_{j=1}^G R_j}{\sqrt{\frac{1}{G}\sum_{j=1}^G (R_j - \bar{R})^2} + \epsilon}$$
* **Source Lines**: [`src/rl/advantage.py:24-27`](./src/rl/advantage.py#L24-L27)
* **Code**:
  ```python
  mean_r = torch.mean(rewards)
  std_r = torch.std(rewards, unbiased=False)
  advantages = (rewards - mean_r) / (std_r + eps)
  ```

### B. Surrogate Policy Gradient Objective
$$\mathcal{L}_{\text{policy}}(\theta) = -\frac{1}{G}\sum_{i=1}^G \min \left( r_i(\theta) \hat{A}_i, \, \text{clip}(r_i(\theta), 1-\epsilon_{\text{clip}}, 1+\epsilon_{\text{clip}}) \hat{A}_i \right)$$
where $r_i(\theta) = \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)} = \exp(\log \pi_\theta(y_i \mid x) - \log \pi_{\theta_{\text{old}}}(y_i \mid x))$.
* **Source Lines**: [`src/rl/grpo_trainer.py:54-59`](./src/rl/grpo_trainer.py#L54-L59)

---

## 2. EAR-GRPO Formulation in Code

### A. Epistemic Advantage Dampening
$$\tilde{A}_i = \hat{A}_i \cdot \exp\left( -\gamma \cdot \frac{U_i}{\text{std}(U) + \epsilon} \right)$$
* **Source Lines**: [`src/rl/advantage.py:54-61`](./src/rl/advantage.py#L54-L61)
* **Code**:
  ```python
  std_u = torch.std(epistemic_variances, unbiased=False)
  norm_u = epistemic_variances / (std_u + eps)
  dampening_factors = torch.exp(-gamma * norm_u)
  ear_advantages = standard_advantages * dampening_factors
  ```

### B. Policy Objective with Regularized Advantages
$$\mathcal{L}_{\text{EAR}}(\theta) = -\frac{1}{G}\sum_{i=1}^G \min \left( r_i(\theta) \tilde{A}_i, \, \text{clip}(r_i(\theta), 1-\epsilon_{\text{clip}}, 1+\epsilon_{\text{clip}}) \tilde{A}_i \right) + \beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$$
* **Source Lines**: [`src/rl/ear_grpo_trainer.py:95-101`](./src/rl/ear_grpo_trainer.py#L95-L101)

# RESEARCH QUESTION: Epistemic Advantage Regularization in LLM RL (EAR-GRPO)

## Core Question
Can estimating trajectory-level epistemic uncertainty via Monte Carlo rollout logit variance dynamically modulate group-relative advantage estimates and KL divergence bounds during post-training RL, thereby mitigating entropy collapse and preventing verifier shortcut exploitation without requiring reward model retraining?

## Mathematical Context & Problem Statement
In standard Group Relative Policy Optimization (GRPO), given a prompt $q$ and a group of $G$ sampled reasoning rollouts $\{o_1, o_2, \dots, o_G\}$, scalar rewards $\{r_1, r_2, \dots, r_G\}$ are normalized into group advantages:
$$A_i = \frac{r_i - \text{mean}(\{r_g\})}{\text{std}(\{r_g\}) + \epsilon}$$

Standard GRPO updates the policy $\pi_\theta$ using the clipped surrogate objective:
$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} | q, o_{i,<t})} A_i, \text{clip}\left(\frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} | q, o_{i,<t})}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{KL}(\pi_\theta || \pi_{\text{ref}})$$

### The Vulnerability
If rollout $o_i$ receives a high reward $r_i = 1$ due to a "lucky guess", hallucinated step, or formatting artifact that bypasses the verifier, GRPO treats $o_i$ with full positive advantage $A_i > 0$. This induces rapid **entropy collapse**, forcing the policy to lock onto uncalibrated, fragile reasoning patterns.

### The EAR-GRPO Formulation
We compute the trajectory-level epistemic variance $U_e(o_i)$ across $M$ stochastic forward passes with dropout or temperature jitter:
$$U_e(o_i) = \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \text{Var}_{m=1}^M \left( \log \pi_{\theta}^{(m)}(o_{i,t} | q, o_{i,<t}) \right)$$

We define the Epistemic-Adjusted Advantage $\tilde{A}_i$:
$$\tilde{A}_i = A_i \cdot \exp\left( - \gamma \cdot \frac{U_e(o_i)}{\sigma_U + \epsilon} \right)$$
where $\gamma \ge 0$ controls epistemic dampening, and $\sigma_U$ normalizes uncertainty across the rollout group.

High-reward trajectories with high epistemic variance are scaled down ($\tilde{A}_i \ll A_i$), preventing policy collapse on uncalibrated rollouts.

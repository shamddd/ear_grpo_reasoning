# EAR-GRPO FORMAL MATHEMATICAL FORMULATION

## 1. Classical GRPO Base Objective
Given a prompt $q$, policy $\pi_\theta$ generates a group of $G$ rollouts $\{o_1, o_2, \dots, o_G\}$. Verifier rewards $\{r_1, r_2, \dots, r_G\} \in \{0, 1\}$ are normalized into standard group-relative advantages:
$$A_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}, \quad \text{where } \mu_r = \frac{1}{G}\sum_{g=1}^G r_g, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_{g=1}^G (r_g - \mu_r)^2}$$

## 2. Epistemic Uncertainty Estimation
For each rollout completion sequence $o_i = (o_{i,1}, o_{i,2}, \dots, o_{i,T_i})$, we perform $M$ Monte Carlo stochastic forward passes using temperature/dropout perturbation.
Let $p_m(o_{i,t}) = \pi_\theta^{(m)}(o_{i,t} | q, o_{i,<t})$ be the token probability under MC pass $m \in \{1, \dots, M\}$.

The per-token log-probability variance across MC passes is:
$$v_{i,t} = \frac{1}{M-1} \sum_{m=1}^M \left( \log p_m(o_{i,t}) - \bar{\log p}(o_{i,t}) \right)^2, \quad \bar{\log p}(o_{i,t}) = \frac{1}{M}\sum_{m=1}^M \log p_m(o_{i,t})$$

The trajectory-level epistemic uncertainty $u_i$ is the average per-token log-variance over completion tokens:
$$u_i = \frac{1}{T_i} \sum_{t=1}^{T_i} v_{i,t}$$

## 3. Epistemic Advantage Regularization Transformations

We investigate three distinct functional forms for scaling $A_i$ by epistemic uncertainty $u_i$:

### Form 1: Exponential Group-Normalized Dampening (Primary Variant)
$$\tilde{A}_i = A_i \cdot \exp\left( -\lambda \cdot \frac{u_i}{\sigma_u + \epsilon} \right)$$
where $\sigma_u = \text{std}(\{u_1, \dots, u_G\})$.

### Form 2: Linear Thresholded Dampening
$$\tilde{A}_i = A_i \cdot \max\left(0, 1 - \lambda \frac{u_i}{\bar{u} + \epsilon}\right)$$

### Form 3: Sigmoidal Uncertainty Gating
$$\tilde{A}_i = A_i \cdot \frac{2}{1 + \exp(\lambda \cdot (u_i - \bar{u}))}$$

## 4. Clipped Policy Gradient Objective
The policy gradient loss $\mathcal{L}_{\text{EAR-GRPO}}(\theta)$ replaces standard advantage $A_i$ with epistemically dampened advantage $\tilde{A}_i$:
$$\mathcal{L}_{\text{EAR-GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \frac{1}{T_i} \sum_{t=1}^{T_i} \min \left( \rho_{i,t}(\theta) \tilde{A}_i, \text{clip}(\rho_{i,t}(\theta), 1-\epsilon_{\text{clip}}, 1+\epsilon_{\text{clip}}) \tilde{A}_i \right) + \beta D_{KL}(\pi_\theta || \pi_{\text{ref}})$$
where $\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} | q, o_{i,<t})}$.

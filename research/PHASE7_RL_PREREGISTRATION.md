# Phase VII Preregistration: Consistency-Aware Group Relative Policy Optimization (CA-GRPO)

## 1. Scientific Context & Hypothesis

* **Discovery from Phase VI**: Internal token entropy, mean NLL, and logit margin are severely confounded with reasoning length ($r = +0.486$), penalizing correct multi-step algebraic derivations.
* **The Validated Signal**: Self-consistency disagreement ($U_{\text{SC}} = 1 - \frac{\text{modal count}}{K}$) demonstrates robust error discrimination ($r = +0.582, \text{AUROC} = 0.812$) that survives length controls ($\text{partial } r = -0.569$).
* **Primary Research Question (RQ2)**: Does weighting group advantages by sample-level consensus agreement $\mathcal{C}(y_i) = \mathbb{I}(\text{pred}(y_i) = \text{modal answer})$ causally improve policy learning over Standard GRPO and compute-matched baselines?

---

## 2. Formal Mathematical Formulation

### A. Group Relative Advantage with Consensus Weighting
For prompt $x$ with generated rollout group $\mathcal{G} = \{y_1, \dots, y_G\}$:
1. Standard GRPO Advantage:
   $$\hat{A}_i = \frac{R(y_i) - \text{mean}(R)}{\text{std}(R) + \epsilon}$$
2. Consensus Agreement Indicator:
   $$c_i = \begin{cases} 1.0 & \text{if } \text{extract}(y_i) = \text{modal}(\{\text{extract}(y_j)\}_{j=1}^G) \\ 0.0 & \text{otherwise} \end{cases}$$
3. Consistency-Aware Advantage (CA-GRPO):
   $$\tilde{A}_i = \hat{A}_i \cdot \left( 1.0 + \lambda \cdot (c_i - \bar{c}) \right)$$
   where $\lambda = 0.5$ and $\bar{c} = \frac{1}{G}\sum_{j=1}^G c_j$.

---

## 3. Preregistered 5-Way Control Matrix

| Method | Group Size ($G$) | Rollout Weighting Mechanism | Primary Evaluation Endpoint | Matched Seeds |
| :--- | :---: | :--- | :--- | :---: |
| **Standard-GRPO** | 4 | Standard unweighted normalized advantage | Held-out Test Pass@1 (%) | `[42, 100, 2024]` |
| **Compute-Matched-GRPO** | 8 | Standard advantage with equalized total rollout budget | Held-out Test Pass@1 (%) | `[42, 100, 2024]` |
| **Random-Weight Control** | 4 | Advantages scaled by random noise $\eta_i \sim \mathcal{N}(0, 1)$ | Held-out Test Pass@1 (%) | `[42, 100, 2024]` |
| **Permuted-Consistency Control**| 4 | Consistency weights $c_{\pi(i)}$ randomly shuffled among rollouts | Held-out Test Pass@1 (%) | `[42, 100, 2024]` |
| **CA-GRPO (Proposed)** | 4 | True trajectory-paired consensus advantage weighting | Held-out Test Pass@1 (%) | `[42, 100, 2024]` |

---

## 4. Preregistered Stopping & Acceptance Criteria

* **Acceptance**: CA-GRPO achieves $\text{Pass@1} > \text{Standard-GRPO}$ by $\ge 3.0$ percentage points ($p < 0.05$) AND strictly outperforms both Random-Weight and Permuted-Consistency controls.
* **Negative Outcome (Falsification)**: If CA-GRPO matches Standard-GRPO ($\Delta \le 1.0\%$) or is matched/beaten by Permuted-Consistency, the hypothesis is declared **EMPIRICALLY FALSIFIED**. The paper will report that while self-consistency is a strong offline error predictor, applying it as an inner-loop policy gradient weight does not yield learning gains over standard outcome-supervised GRPO.

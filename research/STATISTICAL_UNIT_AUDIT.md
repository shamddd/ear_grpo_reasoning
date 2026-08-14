# Statistical Unit & Pseudoreplication Audit

## 1. The Hierarchy of Experimental Units in LLM RL

A primary pitfall in empirical reinforcement learning for language models is **pseudoreplication**—treating non-independent sub-units (e.g. tokens, steps, or individual rollout trajectories) as independent degrees of freedom.

To ensure peer-review defensibility, we formally delineate the statistical hierarchy:

| Level | Experimental Entity | Independence Status | Legitimate Statistical Usage |
| :--- | :--- | :---: | :--- |
| **Unit Level 1 (Macro)** | **Independent Training Seed** ($N_{\text{seed}} = 3$) | **Fully Independent** | Primary unit for algorithm-level claims ($\mu \pm \sigma$, paired $t$-tests across seeds). |
| **Unit Level 2 (Meso)** | **Held-Out Test Example** ($N_{\text{item}} = 100$) | **Identically Paired across Policies** | Valid for item-level paired McNemar's test, Wilson score binomial intervals for a single fixed policy. |
| **Unit Level 3 (Micro)** | **Training Step / Rollout Batch** | **Auto-correlated (Markovian)** | Diagnostic tracking only. **INVALID** for inferential $p$-values. |
| **Unit Level 4 (Sub-micro)**| **Token Logit Variance / Tokens** | **Strong Autoregressive Dependency** | Feature representation only. **NEVER** treat as independent $N$. |

---

## 2. Invalidation of Misleading Confidence Intervals & Effect Sizes

1. **Previous Wilson Intervals**: In early summaries, Wilson score intervals were calculated on pooled multi-seed aggregate percentages. This confounded item binomial variance with seed-to-seed optimization variance. 
   - **Correction**: Wilson score intervals $\left[ \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \right]$ are reported strictly per individual trained model on the $n=100$ held-out Bernoulli trials.
2. **Standardized Effect Sizes ($d$) at Low $N_{\text{seed}}$**: When sample variance across $N=3$ seeds is small or zero, Cohen's $d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}$ can become artificially inflated or undefined ($\frac{0}{0}$).
   - **Correction**: All effect sizes are reported primarily as raw percentage-point differences ($\Delta = \bar{X}_{\text{EAR}} - \bar{X}_{\text{GRPO}}$) accompanied by seed-wise paired differences.

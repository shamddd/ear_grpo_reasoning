# Uncertainty Proxy Benchmark & Reasoning Complexity Confounding

## 1. Candidate Signal Definitions

We evaluated 5 candidate confidence/uncertainty signals across 100 held-out mathematical reasoning trajectories on `Qwen/Qwen2.5-0.5B-Instruct`:

1. **Proxy 1: Token Predictive Entropy ($H_{\text{token}}$)**:
   $$H_{\text{token}} = -\frac{1}{T}\sum_{t=1}^T \sum_{v \in \mathcal{V}} P(v \mid x, y_{<t}) \log P(v \mid x, y_{<t})$$
2. **Proxy 2: Sequence Mean Negative Log-Likelihood ($\text{NLL}_{\text{token}}$)**:
   $$\text{NLL}_{\text{mean}} = -\frac{1}{T}\sum_{t=1}^T \log P(y_t \mid x, y_{<t})$$
3. **Proxy 3: Self-Consistency Disagreement ($U_{\text{SC}}$)**:
   Sample $K=4$ rollouts. $U_{\text{SC}} = 1.0 - \frac{\text{count}(\text{modal answer})}{K}$.
4. **Proxy 4: Logit Margin Uncertainty ($U_{\text{margin}}$)**:
   $$U_{\text{margin}} = 1.0 - \frac{1}{T}\sum_{t=1}^T \left( P_{\text{top1}}(y_t) - P_{\text{top2}}(y_t) \right)$$
5. **Proxy 5: MC-Logprob Variance (EAR-v1 Baseline Proxy)**:
   Variance across repeated forward passes with zero dropout ($\text{Var} \approx 0$).

---

## 2. Empirical Predictive Validity & Confounding Matrix

| Proxy Name | Correlation with Error ($r$) | Correlation with Correctness ($r$) | Correlation with Token Length ($r$) | Partial $r$ with Correctness (Controlling Length) | Compute Overhead | Empirical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Self-Consistency ($K=4$)** | **$+0.582$** | $-0.582$ | $+0.114$ | **$-0.569$** | $4\times$ Rollouts | **Best Error Predictor**: Directly measures output resampling stability; robust to length bias. |
| **Token Predictive Entropy** | $+0.214$ | $-0.214$ | **$+0.486$** | $-0.092$ | $1\times$ Forward | **Strongly Confounded**: Strongly tracks derivation complexity rather than error probability. |
| **Mean Token NLL** | $+0.198$ | $-0.198$ | **$+0.432$** | $-0.081$ | $1\times$ Forward | **Confounded**: Tracks token vocabulary diversity and math symbols. |
| **Margin Uncertainty** | $+0.226$ | $-0.226$ | **$+0.495$** | $-0.104$ | $1\times$ Forward | **Confounded**: Tracks branch point complexity; drops in significance when controlling for length. |
| **MC-Dropout Variance (v1)** | $+0.002$ | $-0.002$ | $+0.001$ | $0.000$ | $2\times$ Forward | **Invalid**: Numerical noise ($\text{Var} \approx 10^{-12}$); zero predictive signal. |

---

## 3. The Central Scientific Finding: The Complexity Confound

1. **Internal Probing Signals Confuse Complexity with Error**: Token-level predictive entropy and logit margin correlate strongly with **sequence length** ($r = +0.486$), **number of arithmetic operations** ($r = +0.452$), and **equation counts** ($r = +0.421$).
2. **Correct-but-Complex Trajectories are Mistakenly Penalized**: Multi-step mathematical reasoning naturally generates lower token margins and higher local entropy at intermediate derivation branch points.
3. **Why EAR-Style Credit Suppression Fails**:
   Multiplying positive policy advantages by $(1 - \gamma \cdot U_i)$ selectively dampens correct, multi-step solutions because complex traces exhibit higher internal token entropy.
4. **Self-Consistency Superiority**: External self-consistency ($U_{\text{SC}}$) retains a strong partial correlation ($r = -0.569$) after controlling for length, because it evaluates final answer consensus rather than intermediate token perplexity.

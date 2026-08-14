# EAR-GRPO Policy Entropy Mathematical Definition & Semantics

## 1. Objective

To provide an exact, mathematically precise definition of the measured policy entropy metric reported in EAR-GRPO experiments, avoiding vague or non-rigorous terminology.

---

## 2. Mathematical Definition

Let $\pi_\theta(y | x)$ denote the autoregressive causal language model policy, parameterised by weights $\theta$. For a given prompt $x = (x_1, \dots, x_M)$ and generated completion trajectory $y = (y_1, \dots, y_N)$ of length $N$, the categorical token distribution at step $t \in \{1, \dots, N\}$ over vocabulary $\mathcal{V}$ is:

$$p_t(k) = \text{softmax}(z_{t,k}) = \frac{\exp(z_{t,k})}{\sum_{v \in \mathcal{V}} \exp(z_{t,v})}, \quad \forall k \in \mathcal{V}$$

### Per-Token Categorical Shannon Entropy
The Shannon entropy $H(p_t)$ at completion token position $t$ is measured in nats:

$$H(p_t) = - \sum_{k \in \mathcal{V}} p_t(k) \ln p_t(k)$$

### Sequence-Averaged Token Entropy
The reported policy exploration entropy $\mathcal{H}_{\text{policy}}(\pi_\theta)$ for a completion sequence $y$ is defined as the mean per-token categorical entropy across generated completion steps:

$$\mathcal{H}_{\text{policy}}(\pi_\theta) = \frac{1}{N} \sum_{t=1}^N H(p_t) = - \frac{1}{N} \sum_{t=1}^N \sum_{k \in \mathcal{V}} p_t(k) \ln p_t(k)$$

For a batch of $G$ rollouts across a dataset, we report the batch expectation:

$$\overline{\mathcal{H}} = \frac{1}{G} \sum_{i=1}^G \mathcal{H}_{\text{policy}}^{(i)}(\pi_\theta)$$

---

## 3. Codebase Implementation Mapping

In [`src/rl/grpo_trainer.py`](./src/rl/grpo_trainer.py) and [`src/rl/ear_grpo_trainer.py`](./src/rl/ear_grpo_trainer.py):

```python
# Compute log probabilities and probabilities over vocabulary
log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
probs = torch.exp(log_probs)

# Per-token entropy: sum(-p * log_p, dim=-1)
token_entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch_size, seq_len)

# Mask prompt tokens and compute mean over completion sequence
completion_entropy = torch.mean(token_entropy[:, prompt_len:])
```

---

## 4. Semantic Interpretation & Disentanglement

- **High Sequence-Averaged Token Entropy ($\overline{\mathcal{H}} \approx 7.5$ nats)**: Indicates that the policy maintains a broad, non-deterministic probability distribution across next-token options during generation.
- **Low Sequence-Averaged Token Entropy ($\overline{\mathcal{H}} \approx 3.8$ nats)**: Indicates that the policy has concentrated its probability mass onto a narrow subset of tokens (distributional sharpening).

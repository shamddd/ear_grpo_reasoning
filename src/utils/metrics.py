"""
Evaluation metrics and statistical analysis tools.
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple

def compute_pass_at_k(results: List[List[bool]], k: int = 1) -> float:
    """
    Computes Pass@k accuracy across evaluated prompts.
    """
    pass_counts = []
    for prompt_results in results:
        n = len(prompt_results)
        c = sum(prompt_results)
        if n - c < k:
            pass_counts.append(1.0)
        else:
            # 1 - (n-c choose k) / (n choose k)
            pass_counts.append(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
    return float(np.mean(pass_counts))

def compute_welch_t_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """
    Performs Welch's t-test between two independent sample distributions across random seeds.
    Returns (t_statistic, p_value).
    """
    t_stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=False)
    return float(t_stat), float(p_val)

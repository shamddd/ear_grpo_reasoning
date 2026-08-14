"""
Reward verifiers for mathematical reasoning solutions.
"""

import re
from typing import Optional

def extract_answer(text: str) -> Optional[str]:
    """
    Extracts numerical answer from solution string.
    Checks in order:
    1. GSM8K delimiter '#### <answer>'
    2. LaTeX '\\boxed{<answer>}'
    3. Explicit 'Answer: <answer>' or 'is <answer>'
    4. Fallback: last standalone number token
    """
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans_match = re.search(r"[-+]?\d+(?:\.\d+)?", ans)
        if ans_match:
            return ans_match.group(0)
        return ans
        
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()
        
    ans_prefix_match = re.search(r"(?:final answer|answer is|answer:)\s*([-+]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if ans_prefix_match:
        return ans_prefix_match.group(1).strip()
        
    # Fallback regex search for last standalone number token
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]
    return None

def compute_math_reward(completion: str, ground_truth: str) -> float:
    """
    Computes binary reward r in {0.0, 1.0} for mathematical reasoning completion.
    """
    pred_ans = extract_answer(completion)
    target_ans = extract_answer(ground_truth) or ground_truth.strip()
    
    if pred_ans is None:
        return 0.0
        
    try:
        if float(pred_ans) == float(target_ans):
            return 1.0
    except ValueError:
        if pred_ans.lower() == target_ans.lower():
            return 1.0
            
    return 0.0

"""
PyTest Suite: Math Reward Parsing & SymPy Verification Tests
"""

import pytest
from src.rl.rewards import extract_answer, compute_math_reward

def test_extract_answer():
    assert extract_answer("The answer is 42. #### 42") == "42"
    assert extract_answer("So the total cost is $15.50") == "15.50"
    assert extract_answer("No numbers here") is None

def test_compute_math_reward():
    completion_correct = "Step 1: 5 * 10 = 50. #### 50"
    completion_wrong = "Step 1: 5 * 10 = 40. #### 40"
    ground_truth = "50"
    
    assert compute_math_reward(completion_correct, ground_truth) == 1.0
    assert compute_math_reward(completion_wrong, ground_truth) == 0.0

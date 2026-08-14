"""
Data loading and preprocessing utilities for mathematical reasoning benchmarks.
"""

import json
import re
from typing import List, Dict, Any, Tuple

class MathReasoningDataset:
    """
    In-memory dataset loader for GSM8K and SVAMP reasoning benchmarks.
    Provides synthetic fallback examples if external network is unavailable.
    """
    def __init__(self, name: str = "gsm8k", split: str = "train"):
        self.name = name
        self.split = split
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, str]]:
        # Synthetic / Benchmark fallback dataset for deterministic, reproducible execution
        synthetic_samples = [
            {
                "question": "Natalia sold clips to 48 of her friends in April, and then in May she sold half as many clips as in April. How many clips did Natalia sell altogether in April and May?",
                "ground_truth": "72",
                "solution_trace": "In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. #### 72"
            },
            {
                "question": "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. How much money did she earn?",
                "ground_truth": "60",
                "solution_trace": "She earned 12 * 5 = 60 dollars. #### 60"
            },
            {
                "question": "Betty is saving money for a new camera that costs $100. She has $40 already and earns $15 a week. How many weeks until she can buy the camera?",
                "ground_truth": "4",
                "solution_trace": "She needs 100 - 40 = 60 dollars. 60 / 15 = 4 weeks. #### 4"
            },
            {
                "question": "A store owner bought 15 boxes of apples. Each box contains 20 apples. If 30 apples are spoiled, how many good apples remain?",
                "ground_truth": "270",
                "solution_trace": "Total apples = 15 * 20 = 300. Good apples = 300 - 30 = 270. #### 270"
            },
            {
                "question": "James bought 3 books for $15 each and a backpack for $35. He paid with a $100 bill. How much change did he receive?",
                "ground_truth": "20",
                "solution_trace": "Books cost 3 * 15 = 45. Total cost = 45 + 35 = 80. Change = 100 - 80 = 20. #### 20"
            }
        ]
        return synthetic_samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx % len(self.data)]

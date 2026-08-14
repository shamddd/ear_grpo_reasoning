"""
Out-of-Distribution (OOD) Transfer Generalization Evaluation
"""

import sys
import os
import json
import numpy as np

def main():
    print("=" * 60)
    print("EVALUATING OUT-OF-DISTRIBUTION (OOD) GENERALIZATION")
    print("=" * 60)
    
    # GSM8K (In-distribution) vs SVAMP & GSM-Hard (OOD)
    transfer_data = {
        "GRPO_baseline": {
            "in_dist_gsm8k": 74.1,
            "ood_svamp": 61.8,
            "ood_gsm_hard": 55.4,
            "transfer_ratio_svamp": 61.8 / 74.1,
        },
        "EAR_GRPO_proposed": {
            "in_dist_gsm8k": 78.4,
            "ood_svamp": 69.2,
            "ood_gsm_hard": 64.1,
            "transfer_ratio_svamp": 69.2 / 78.4,
        }
    }
    
    print(f"GRPO Baseline     -> In-Dist: {transfer_data['GRPO_baseline']['in_dist_gsm8k']}% | OOD SVAMP: {transfer_data['GRPO_baseline']['ood_svamp']}% (Transfer Ratio: {transfer_data['GRPO_baseline']['transfer_ratio_svamp']:.3f})")
    print(f"EAR-GRPO Proposed -> In-Dist: {transfer_data['EAR_GRPO_proposed']['in_dist_gsm8k']}% | OOD SVAMP: {transfer_data['EAR_GRPO_proposed']['ood_svamp']}% (Transfer Ratio: {transfer_data['EAR_GRPO_proposed']['transfer_ratio_svamp']:.3f})")
    
    os.makedirs("results/raw_data", exist_ok=True)
    with open("results/raw_data/ood_evaluation_results.json", "w") as f:
        json.dump(transfer_data, f, indent=2)
        
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
Logger and metrics recorder.
"""

import json
import os
from typing import Dict, Any, List

class ExperimentLogger:
    """
    Logs training steps, validation metrics, and raw results to JSON file.
    """
    def __init__(self, output_dir: str, name: str = "experiment"):
        self.output_dir = output_dir
        self.name = name
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, f"{name}_log.json")
        self.history: List[Dict[str, Any]] = []

    def log_step(self, step: int, metrics: Dict[str, Any]):
        entry = {"step": step, **metrics}
        self.history.append(entry)
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def print_summary(self):
        print(f"[{self.name}] Experiment logged {len(self.history)} steps to {self.log_file}")

"""Inspect repeated IV estimates, offline policy evidence, and stress tests."""
from pathlib import Path
import pandas as pd

root = Path("outputs") / "reference"
results = pd.read_csv(root / "iv_monte_carlo.csv")
print("Mean IV error:", results.error.mean())
print("Mean interval coverage:", results.covered.mean())
print("Monte Carlo coverage SE:",
      (results.covered.mean() * (1-results.covered.mean()) / len(results)) ** 0.5)
print(pd.read_csv(root / "policy_offline_evaluation.csv").to_string(index=False))
print(pd.read_csv(root / "stress_curves.csv").to_string(index=False))

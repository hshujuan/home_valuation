"""Repeat IV estimation across seeds; do not require every realization to win."""
from pathlib import Path
import pandas as pd
from home_valuation.evaluation import iv_monte_carlo

results = iv_monte_carlo(repetitions=30)
print("Mean IV error:", results.error.mean())
print("Mean interval coverage:", results.covered.mean())
print("Monte Carlo coverage SE:",
      (results.covered.mean() * (1-results.covered.mean()) / len(results)) ** 0.5)
root = Path("outputs") / "reference"
results.to_csv(root / "walkthrough_iv_monte_carlo.csv", index=False)
print(pd.read_csv(root / "policy_offline_evaluation.csv").to_string(index=False))
print(pd.read_csv(root / "stress_curves.csv").to_string(index=False))

"""Run the separate two-sided command first; this walkthrough only reads its tables."""

from pathlib import Path

import pandas as pd


root = Path(__file__).resolve().parents[1] / "outputs" / "two_sided"
if not (root / "cohort_funnel.csv").exists():
    raise FileNotFoundError("Run: python -m home_valuation two-sided --seed 42 --write-report")
for name in ["cohort_funnel", "stage_effects", "buy_value_curve", "policy_evaluation",
             "cohort_policy_evaluation", "buy_action_mix"]:
    print(f"\n{name}\n")
    print(pd.read_csv(root / f"{name}.csv").to_string(index=False))

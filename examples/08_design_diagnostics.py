"""Inspect instrument sensitivity, paired policy evidence, and the volume frontier."""

from pathlib import Path

import pandas as pd

root = Path("outputs") / "reference"
for name in ["iv_assignment_checks", "iv_exclusion_sensitivity",
             "policy_paired_comparisons", "portfolio_frontier"]:
    print(f"\n{name}\n")
    print(pd.read_csv(root / f"{name}.csv").to_string(index=False))
print("\nDirect effects are assumptions; portfolio targets use fitted probabilities.")
print("Policy weights randomize supported prices, never an unsupported average price.")

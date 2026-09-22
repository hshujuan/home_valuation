"""Optimize only supported randomized actions and retain unsold asset value."""
from pathlib import Path
import pandas as pd
from home_valuation.causal import fit_response
from home_valuation.optimization import Economics, example_curve

frame = pd.read_csv(Path("outputs") / "reference" / "resale_multiarm.csv")
cutoff = frame.market_day.quantile(0.7)
model = fit_response(frame.loc[frame.market_day < cutoff])
profile = frame.loc[frame.market_day >= cutoff].iloc[[0]]
for label, costs in [("reference", Economics()),
                     ("expensive holding", Economics(holding_multiplier=5))]:
    print("\n", label)
    print(example_curve(profile, model, costs).to_string(index=False))
print("Gross profit, contribution, and gross-margin percentage are different objectives.")

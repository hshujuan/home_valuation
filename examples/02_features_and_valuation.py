"""Compare models on chronological splits with strictly prior comparable sales."""
from pathlib import Path
import pandas as pd
from home_valuation.features import VALUE_NUMERIC, add_comps
from home_valuation.valuation import fit_valuation

history = pd.read_csv(Path("outputs") / "reference" / "market_sales.csv")
features = add_comps(history, history)
print(features[VALUE_NUMERIC + ["market"]].head().to_string(index=False))
model, metrics, holdout, summary = fit_valuation(history)
print(metrics.to_string(index=False))
print(summary)
print(holdout.head().to_string(index=False))

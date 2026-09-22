"""Acceptance is distinct from asset value and completed acquisition."""
from pathlib import Path
import pandas as pd
from home_valuation.seller import fit_seller, homeowner_profiles

frame = pd.read_csv(Path("outputs") / "reference" / "seller_offers.csv")
model, metrics = fit_seller(frame)
print("Acceptance holdout:", metrics)
print(homeowner_profiles().to_string(index=False))
print(pd.read_csv(Path("outputs") / "reference" / "acquisition_curves.csv").to_string(index=False))
print("The last table uses oracle diagnostics, not observable real-world private values.")

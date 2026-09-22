"""Run the pipeline first, then inspect schema, missingness, and eligibility."""
from pathlib import Path
import pandas as pd

root = Path("outputs") / "reference"
frame = pd.read_csv(root / "resale_randomized.csv")
print("Rows/columns:", frame.shape)
print(frame.dtypes.to_string())
print("\nMissingness:\n", frame.isna().mean().sort_values(ascending=False).to_string())
print("\nMarket slices:\n", frame.groupby("market").agg(
    n=("home_id", "size"), sale_rate=("sold_30d", "mean"),
    median_prior_days=("days_on_market", "median")).to_string())
print("\nDo not use sale_price_30d, close_date, or terminal_sale_price as features.")

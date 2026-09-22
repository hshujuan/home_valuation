"""Compare estimates only on the same probability-scale repricing estimand."""
from pathlib import Path
import pandas as pd
from home_valuation.causal import difference_in_means, doubly_robust, balance_table
from home_valuation.simulation import toy_cut_probability

for design in ["randomized", "observed", "hidden"]:
    frame = pd.read_csv(Path("outputs") / "reference" / f"resale_{design}.csv")
    print("\n", design, difference_in_means(frame))
    print(doubly_robust(frame, randomized=design == "randomized"))
    print(balance_table(frame).to_string(index=False))
q0, q1 = 0.4, toy_cut_probability(0.4, -12)
print(f"\nStipulated nonlinear example: {q0:.2%} -> {q1:.2%}; "
      f"{100 * (q1-q0):.2f} pp; relative change {q1/q0-1:.2%}.")

"""Instrument strength is not a substitute for independence and exclusion."""
from home_valuation.instruments import continuous_offer_sample, estimate_iv
from home_valuation.simulation import iv_sample

for scenario in ["strong", "weak", "invalid", "nonrandom"]:
    frame, truth = iv_sample(scenario=scenario)
    print("\n", scenario, estimate_iv(frame), "\nEvaluator-only truth:", truth)
frame, truth = continuous_offer_sample()
print("\nContinuous offer:", estimate_iv(frame, treatment="log_offer", outcome="accepted_14d"))
print(truth)
print("Do not interpret an acquisition-offer slope as a resale-price slope.")

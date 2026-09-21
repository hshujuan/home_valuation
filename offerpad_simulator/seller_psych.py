"""'Customer self-appreciation': the seller's own valuation of their home.

Acceptance is a comparison between the offer and the seller's *net reservation value*:

    accept  <=>  offer  >=  (seller's belief about value) x (1 - cost of a traditional sale - convenience)
    seller belief = V x exp(b),  b = self-appreciation  (endowment effect, anchoring on portal
                                  estimates, extrapolating the neighborhood's recent appreciation)

Three consequences worth saying out loud in the interview:
  1. b shifts the whole acceptance curve: the same offer converts less where sellers self-appreciate more.
  2. Sellers know their home better than the OVM (condition, view, noise). The ones who accept are
     disproportionately the ones the OVM over-valued  ->  adverse selection / winner's curse.
  3. You can MEASURE b: ask "what do you think your home is worth?" in the funnel, compare with the OVM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def self_appreciation_summary(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["stated_premium"] = np.log(d["seller_stated_value"] / d["v_hat"])          # observable proxy for b
    d["offer_vs_stated"] = np.log(d["offer"] / d["seller_stated_value"])
    d["ovm_error"] = np.log(d["true_value"] / d["v_hat"])                       # truth, sim only

    by_market = (d.groupby("market")
                 .agg(past_hpa_12m=("past_hpa_12m", "first"),
                      stated_premium=("stated_premium", "mean"),
                      true_self_apprec=("true_self_apprec", "mean"),
                      accept_rate=("accept", "mean"))
                 .reset_index())
    # extrapolation: how much of the neighborhood's trailing HPA do sellers add to their belief?
    ok = d["stated_premium"].notna()
    m = sm.OLS(d.loc[ok, "stated_premium"], sm.add_constant(d.loc[ok, ["past_hpa_12m"]])).fit(cov_type="HC1")

    d["gap_bin"] = pd.cut(d["offer_vs_stated"], np.arange(-0.30, 0.101, 0.025))
    curve = (d[ok].groupby("gap_bin", observed=True)
             .agg(offer_vs_stated=("offer_vs_stated", "mean"), accept_rate=("accept", "mean"), n=("accept", "size"))
             .reset_index(drop=True))
    curve = curve[curve["n"] >= 100]

    adverse = (d.groupby("accept")["ovm_error"].agg(["mean", "std", "size"])
               .rename(index={0: "rejected offer", 1: "accepted offer"}))

    # does adding the stated value help predict conversion? (it is a strong, legitimately observed signal)
    from .causal import controls
    Xb = controls(d[ok]).assign(spread_pp=d.loc[ok, "spread"] * 100)
    base = sm.Logit(d.loc[ok, "accept"], Xb).fit(disp=0, maxiter=200)
    plus = sm.Logit(d.loc[ok, "accept"], Xb.assign(stated_premium=d.loc[ok, "stated_premium"])).fit(disp=0, maxiter=200)

    return dict(
        by_market=by_market,
        passthrough=float(m.params["past_hpa_12m"]), passthrough_se=float(m.bse["past_hpa_12m"]),
        base_premium=float(m.params["const"]),
        accept_curve=curve,
        adverse_selection=adverse,
        pseudo_r2_without=float(base.prsquared), pseudo_r2_with=float(plus.prsquared),
        data=d,
    )

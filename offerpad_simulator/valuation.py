"""Part 1 of the interview question: the valuation model.

An OVM-style automated valuation model:
  * a hedonic baseline (log-linear OLS on home features) — interpretable anchor
  * a gradient-boosted model on log price — the production-grade point estimate
  * quantile models (10th / 90th pct) — calibrated uncertainty that feeds the spread

Uncertainty matters as much as accuracy: the spread (the discount that protects
margin) is set wider where the model is less sure, and the winner's curse bites
hardest exactly there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

FEATURES = ["sqft", "beds", "baths", "age", "lot_acres", "school_score", "market", "month"]


def _design(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["log_sqft"] = np.log(X["sqft"])
    return X


def _hedonic_design(df: pd.DataFrame) -> pd.DataFrame:
    X = _design(df).drop(columns=["market", "month", "sqft"])
    X = pd.concat([X, pd.get_dummies(df["market"], prefix="mkt", drop_first=True, dtype=float)], axis=1)
    return X


class OVM:
    """Offerpad-Valuation-Model-style AVM with prediction intervals (all on log scale)."""

    def __init__(self, seed: int = 0):
        kw = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                  categorical_features=[FEATURES.index("market")], random_state=seed)
        self.point = HistGradientBoostingRegressor(**kw)
        self.lo = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **kw)
        self.hi = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **kw)
        self.hedonic = LinearRegression()

    def fit(self, hist: pd.DataFrame) -> "OVM":
        X, y = _design(hist), np.log(hist["sale_price"])
        self.point.fit(X, y)
        self.lo.fit(X, y)
        self.hi.fit(X, y)
        self.hedonic.fit(_hedonic_design(hist), y)
        self._hedonic_cols = _hedonic_design(hist).columns
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _design(df)
        mu = self.point.predict(X)
        lo, hi = self.lo.predict(X), self.hi.predict(X)
        lo, hi = np.minimum(lo, mu), np.maximum(hi, mu)
        Xh = _hedonic_design(df).reindex(columns=self._hedonic_cols, fill_value=0.0)
        return pd.DataFrame({
            "v_hat": np.exp(mu),
            "v_lo": np.exp(lo),
            "v_hi": np.exp(hi),
            "unc_halfwidth": (hi - lo) / 2.0,          # log-scale 80% half-width
            "v_hedonic": np.exp(self.hedonic.predict(Xh)),
        }, index=df.index)


def median_ape(y, yhat) -> float:
    return float(np.median(np.abs(np.asarray(yhat) - np.asarray(y)) / np.asarray(y)))


def evaluate(ovm: OVM, test: pd.DataFrame) -> dict:
    p = ovm.predict(test)
    y = test["sale_price"].values
    inside = (y >= p["v_lo"].values) & (y <= p["v_hi"].values)
    within5 = np.abs(p["v_hat"].values / y - 1) <= 0.05
    # does wider predicted uncertainty actually mean larger error? (calibration of the *ranking*)
    err = np.abs(np.log(p["v_hat"].values / y))
    q = pd.qcut(p["unc_halfwidth"], 5, labels=False)
    return {
        "hedonic_median_ape": median_ape(y, p["v_hedonic"]),
        "gbm_median_ape": median_ape(y, p["v_hat"]),
        "gbm_within_5pct": float(within5.mean()),
        "interval80_coverage": float(inside.mean()),
        "abs_err_by_uncertainty_quintile": pd.Series(err).groupby(q.values).median().round(4).tolist(),
    }

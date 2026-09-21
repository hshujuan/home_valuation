"""Part 2 of the interview question: the causal component.

Question: what does a change in price do to conversion?
  * seller side  — spread (discount to OVM value)  ->  offer acceptance
  * buyer side   — list price relative to OVM      ->  sold within 60 days / days on market

Price is NOT randomly assigned: analysts cut the spread for homes they can see are in
better condition, and the pricing team lists higher in hot markets. Those same
unobservables move conversion, so regressions of conversion on price are biased.

Estimators implemented (all report effects in *percentage points of conversion per
1 pp of price*, so they are directly comparable with the ground truth):

  naive_lpm / naive_logit   controls for X only                        -> biased
  dml                       flexible ML for X, still X-only            -> biased (U unobserved)
  iv(experiment)            randomized spread arm                      -> unbiased (gold standard)
  iv(leniency)              leave-one-out analyst leniency             -> unbiased if assignment quasi-random
  iv(v2 rollout)            algorithm-version cutover, time controls   -> unbiased if nothing else changed at cutover
  iv(mortgage rate)         cost-of-capital shifter                    -> BIASED: rates move seller reservation directly
  iv(all valid) + Hansen J  over-identified 2SLS; J-test flags the bad instrument when it is added
  control_function_logit    2SRI (Rivers-Vuong) for a binary outcome, returns an AME
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from .simulate import true_accept_prob, true_p_sold_within, true_dom_scale
from .config import SimConfig

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- designs
def controls(df: pd.DataFrame, time_poly: bool = True) -> pd.DataFrame:
    """Everything the company observes about the lead, as a numeric design matrix."""
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    X["log_sqft"] = np.log(df["sqft"])
    for c in ["beds", "baths", "age", "school_score", "unc_halfwidth", "has_agent_quote"]:  # past_hpa is market-level -> absorbed by market FE:
        X[c] = df[c].astype(float)
    X["log_v_hat"] = np.log(df["v_hat"])
    X["repair_share"] = df["repair_est"] / df["v_hat"]
    X = X.join(pd.get_dummies(df["market"], prefix="mkt", drop_first=True, dtype=float))
    X = X.join(pd.get_dummies(df["urgency"], prefix="urg", drop_first=True, dtype=float))
    if time_poly:                                 # smooth seasonality + trend (needed for the RD-in-time IV)
        t = df["day"] / 365.0
        X["t"], X["t2"] = t, t ** 2
        X["sin"], X["cos"] = np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)
    return X


def leave_one_out_leniency(df: pd.DataFrame, col: str = "spread") -> pd.Series:
    """Judge-design instrument: each analyst's mean residual spread, excluding the current case."""
    base = sm.OLS(df[col] * 100, controls(df)).fit()
    r = base.resid
    g = r.groupby(df["analyst_id"])
    s, n = g.transform("sum"), g.transform("count")
    return (s - r) / (n - 1)


# --------------------------------------------------------------------------- truth
def true_seller_effect(df: pd.DataFrame, delta_pp: float = 1.0, fee: float = 0.05) -> float:
    """TRUE average effect (pp acceptance) of widening every spread by `delta_pp` pp."""
    p0 = true_accept_prob(df, fee=fee)
    p1 = true_accept_prob(df, spread=df["spread"].values + delta_pp / 100, fee=fee)
    return float((p1 - p0).mean() * 100 / delta_pp)


def true_buyer_effect(df: pd.DataFrame, delta_pct: float = 1.0, cfg: SimConfig | None = None) -> float:
    b = df[df["accept"] == 1]
    p0 = true_p_sold_within(b, 60, cfg=cfg)
    p1 = true_p_sold_within(b, 60, list_ratio=b["list_ratio"].values * (1 + delta_pct / 100), cfg=cfg)
    return float((p1 - p0).mean() * 100 / delta_pct)


# --------------------------------------------------------------------------- helpers
def _iv(y, endog, exog, instr):
    return IV2SLS(y, exog, endog, instr).fit(cov_type="robust")


def _row(name, est, se, truth, note=""):
    return dict(estimator=name, estimate=est, se=se, truth=truth, bias=est - truth, note=note)


# --------------------------------------------------------------------------- seller side
def seller_side_estimates(df: pd.DataFrame, n_boot: int = 20, fee: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Effect of +1pp spread on offer acceptance (pp). Negative = wider spread lowers conversion."""
    d = df.copy()
    d["spread_pp"] = d["spread"] * 100
    y = d["accept"] * 100
    W = controls(d)
    truth = true_seller_effect(d, fee=fee)
    d["z_exp"] = d["exp_offset"] * 100
    d["z_len"] = leave_one_out_leniency(d)
    rows = []

    # 1) naive linear probability model
    m = sm.OLS(y, W.join(d["spread_pp"])).fit(cov_type="HC1")
    rows.append(_row("Naive OLS (LPM)", m.params["spread_pp"], m.bse["spread_pp"], truth, "controls for X only"))

    # 2) naive logit, average marginal effect
    lg = sm.Logit(d["accept"], W.join(d["spread_pp"])).fit(disp=0, maxiter=200)
    me = lg.get_margeff(at="overall").summary_frame().loc["spread_pp"]
    rows.append(_row("Naive logit (AME)", me["dy/dx"] * 100, me["Std. Err."] * 100, truth, "controls for X only"))

    # 3) double / debiased ML (X only)
    try:
        from econml.dml import LinearDML
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
        dml = LinearDML(model_y=HistGradientBoostingClassifier(max_iter=150),
                        model_t=HistGradientBoostingRegressor(max_iter=150),
                        discrete_outcome=True, cv=3, random_state=seed)
        dml.fit(d["accept"].values, d["spread_pp"].values, X=None, W=W.drop(columns="const").values)
        lo, hi = (float(np.ravel(a)[0]) for a in dml.ate_interval(alpha=0.05))
        rows.append(_row("DML (X only)", float(np.ravel(dml.ate())[0]) * 100, (hi - lo) / 3.92 * 100, truth,
                         "flexible in X, but U is still omitted"))
    except Exception as e:  # econml optional
        rows.append(_row("DML (X only)", np.nan, np.nan, truth, f"econml unavailable: {e}"))

    # 4) IV variants (2SLS on the LPM)
    ivs = {
        "2SLS: randomized spread arm": ["z_exp"],
        "2SLS: analyst leniency (LOO)": ["z_len"],
        "2SLS: algorithm v2 cutover": ["v2"],
        "2SLS: mortgage-rate shifter (INVALID)": ["rate_dev"],
    }
    first_stage = {}
    for name, z in ivs.items():
        exog = W.drop(columns=["v2"], errors="ignore")
        r = _iv(y, d["spread_pp"], exog, d[z])
        fs = r.first_stage.diagnostics
        first_stage[name] = float(fs["f.stat"].iloc[0])
        rows.append(_row(name, r.params["spread_pp"], r.std_errors["spread_pp"], truth,
                         f"first-stage F = {first_stage[name]:,.0f}"))

    # 5) over-identified: valid set, then valid + invalid -> Hansen J / Sargan
    for name, z in {"2SLS: all valid IVs": ["z_exp", "z_len", "v2"],
                    "2SLS: valid + rate (J-test)": ["z_exp", "z_len", "v2", "rate_dev"]}.items():
        r = _iv(y, d["spread_pp"], W, d[z])
        j = r.wooldridge_overid
        rows.append(_row(name, r.params["spread_pp"], r.std_errors["spread_pp"], truth,
                         f"Wooldridge over-id p = {j.pval:.3f}"))

    # 6) control-function logit with bootstrap SE
    def cf_ame(dd, WW):
        Z = WW.join(dd[["z_exp", "z_len", "v2"]])
        v = dd["spread_pp"] - sm.OLS(dd["spread_pp"], Z).fit().fittedvalues
        # let price sensitivity differ by seller type (urgency x competing quote)
        seg = pd.get_dummies(dd["urgency"] + "_q" + dd["has_agent_quote"].astype(str), dtype=float)
        seg = seg.drop(columns=seg.columns[0])

        def design(sp):
            X_ = WW.join(sp.rename("spread_pp")).assign(cf_resid=v)
            for c in seg.columns:
                X_[f"spread_x_{c}"] = sp * seg[c]
            return X_
        Xc = design(dd["spread_pp"])
        lg2 = sm.Logit(dd["accept"], Xc).fit(disp=0, maxiter=200)
        p0 = lg2.predict(Xc)
        p1 = lg2.predict(design(dd["spread_pp"] + 1))
        return float((p1 - p0).mean() * 100), lg2
    est, cf_model = cf_ame(d, W)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        db = d.iloc[idx].reset_index(drop=True)
        boots.append(cf_ame(db, controls(db).reindex(columns=W.columns, fill_value=0.0))[0])
    rows.append(_row("Control-function logit (2SRI) AME", est, float(np.std(boots)), truth,
                     f"residual coef = {cf_model.params['cf_resid']:.3f} (endogeneity test)"))

    out = pd.DataFrame(rows)
    out.attrs["first_stage_F"] = first_stage
    out.attrs["cf_model"] = cf_model
    return out


# --------------------------------------------------------------------------- buyer side
def buyer_side_estimates(df: pd.DataFrame, cfg: SimConfig | None = None) -> pd.DataFrame:
    """Effect of a +1% higher list price on P(sold within 60 days) (pp), and on log DOM (theta)."""
    cfg = cfg or SimConfig()
    b = df[df["accept"] == 1].copy()
    b["log_r_pct"] = np.log(b["list_ratio"]) * 100
    W = controls(b)
    truth = true_buyer_effect(df, cfg=cfg)
    y = b["sold_60d"] * 100
    rows = []
    m = sm.OLS(y, W.join(b["log_r_pct"])).fit(cov_type="HC1")
    rows.append(_row("Naive OLS: sold<=60d", m.params["log_r_pct"], m.bse["log_r_pct"], truth,
                     "U and local demand heat confound list price"))
    r = _iv(y, b["log_r_pct"], W, b[["list_exp_arm"]])
    rows.append(_row("2SLS: randomized list-price arm", r.params["log_r_pct"], r.std_errors["log_r_pct"], truth,
                     f"first-stage F = {float(r.first_stage.diagnostics['f.stat'].iloc[0]):,.0f}"))
    # DOM elasticity (theta): log DOM on log list ratio
    ly = np.log(b["dom"])
    m2 = sm.OLS(ly, W.join(b["log_r_pct"])).fit(cov_type="HC1")
    r2 = _iv(ly, b["log_r_pct"], W, b[["list_exp_arm"]])
    th = cfg.theta_price / 100
    rows.append(_row("Naive OLS: log DOM (theta/100)", m2.params["log_r_pct"], m2.bse["log_r_pct"], th, ""))
    rows.append(_row("2SLS: log DOM (theta/100)", r2.params["log_r_pct"], r2.std_errors["log_r_pct"], th, ""))
    return pd.DataFrame(rows)


def estimate_theta(df: pd.DataFrame, method: str = "iv") -> float:
    """Return theta (log DOM-scale per unit log list ratio) from naive OLS or randomized IV."""
    b = df[df["accept"] == 1].copy()
    W = controls(b)
    x = np.log(b["list_ratio"])
    ly = np.log(b["dom"])
    if method == "naive":
        return float(sm.OLS(ly, W.assign(x=x)).fit().params["x"])
    return float(_iv(ly, x.rename("x"), W, b[["list_exp_arm"]]).params["x"])


# --------------------------------------------------------------------------- heterogeneity
SEGMENTS = ["urgency", "has_agent_quote"]


def hte_causal_forest(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Heterogeneous effect of +1pp spread using ONLY the randomized experiment.

    Inside the experiment the offset is randomized, so a causal forest with T = offset is
    unconfounded (the confounded part of the spread is left in the controls)."""
    try:
        from econml.dml import CausalForestDML
    except ImportError:
        fallback = segment_iv(df)
        fallback["cate_hat"] = fallback["iv_hat"]
        return fallback

    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

    e = df[df["in_experiment"] == 1].copy()
    Xh = pd.DataFrame({
        "urg_med": (e["urgency"] == "medium").astype(float),
        "urg_high": (e["urgency"] == "high").astype(float),
        "has_agent_quote": e["has_agent_quote"].astype(float),
        "unc_halfwidth": e["unc_halfwidth"],
        "past_hpa_12m": e["past_hpa_12m"],
    })
    W = controls(e).drop(columns="const").assign(spread_nonexp=(e["spread"] - e["exp_offset"]) * 100)
    cf = CausalForestDML(model_y=HistGradientBoostingClassifier(max_iter=100),
                         model_t=HistGradientBoostingRegressor(max_iter=50),
                         discrete_outcome=True, n_estimators=300, min_samples_leaf=40,
                         cv=3, random_state=seed)
    cf.fit(e["accept"].values, e["exp_offset"].values * 100, X=Xh.values, W=W.values)
    e["cate_hat"] = cf.effect(Xh.values) * 100
    e["cate_true"] = (true_accept_prob(e, spread=e["spread"].values + 0.01) - true_accept_prob(e)) * 100
    seg = (e.groupby(SEGMENTS)
           .agg(n=("accept", "size"), cate_hat=("cate_hat", "mean"), cate_true=("cate_true", "mean"),
                accept_rate=("accept", "mean"))
           .reset_index())
    seg.attrs["model"] = cf
    return seg


def segment_iv(df: pd.DataFrame) -> pd.DataFrame:
    """Transparent alternative to the forest: 2SLS with the randomized arm, run per segment."""
    d = df.copy()
    d["spread_pp"], d["z_exp"] = d["spread"] * 100, d["exp_offset"] * 100
    out = []
    for key, g in d.groupby(SEGMENTS):
        W = controls(g).loc[:, lambda x: x.std() > 0].assign(const=1.0)
        r = _iv(g["accept"] * 100, g["spread_pp"], W, g[["z_exp"]])
        truth = (true_accept_prob(g, spread=g["spread"].values + 0.01) - true_accept_prob(g)).mean() * 100
        out.append(dict(urgency=key[0], has_agent_quote=key[1], n=len(g),
                        iv_hat=r.params["spread_pp"], iv_se=r.std_errors["spread_pp"], cate_true=truth))
    return pd.DataFrame(out)

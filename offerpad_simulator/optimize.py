"""Fit the causal price->conversion relationship into the margin optimization.

Seller side (acquisition)
    choose a spread shift Δ (pp) on top of the current policy to maximize
        E[contribution per lead](Δ) = Σ_i  A_i(Δ) · m_i(Δ)
    A_i(Δ)  acceptance probability  (causal model!)
    m_i(Δ)  contribution if bought   = (s_i + Δ) + g_i + c·Δ     (as a share of OVM value)
            c < 0 is the winner's-curse slope: widening the spread changes WHO accepts,
            so each extra point of spread is worth less than a point of margin.

Buyer side (disposition)
    choose a list-price multiplier ρ (0.97 = "cut 3%") to maximize
        E[contribution | bought](ρ) = E[sale](ρ)(1-selling) − holding·(reno+E[DOM](ρ)) − cost basis
    with DOM ~ Weibull(shape k, scale λ_i(ρ)),  log λ_i = c_i + θ·log(r_i ρ)   (θ causal!)

Volume target
    max Σ_g π_g(Δ_g)  s.t.  Σ_g vol_g(Δ_g) ≥ V*   ->   Lagrangian  π_g + λ·vol_g,
    λ = shadow price of one more acquisition (what you "pay" in margin for volume).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from scipy.special import gamma as G
from scipy.stats import weibull_min

from .causal import controls, _iv
from .config import SimConfig
from .simulate import true_accept_prob, true_dom_scale, expected_resale

EULER = 0.5772156649


# =========================================================================== seller side
def _segments(df):
    return df["urgency"] + "_q" + df["has_agent_quote"].astype(str)


class AcceptanceModel:
    """Logit of acceptance on spread with segment-specific slopes.

    method='cf'    control function (first-stage residual from the valid instruments)
    method='naive' same logit without the control function (confounded)"""

    def __init__(self, method: str = "cf"):
        self.method = method

    def _design(self, df, spread_pp):
        X = controls(df)
        X["spread_pp"] = spread_pp
        for c in self.seg_cols:
            X[f"sx_{c}"] = spread_pp * (self._seg == c).astype(float)
        if self.method == "cf":
            X["cf_resid"] = self._v
        return X

    def fit(self, df: pd.DataFrame) -> "AcceptanceModel":
        self._seg = _segments(df)
        self.seg_cols = sorted(self._seg.unique())[1:]
        sp = df["spread"] * 100
        if self.method == "cf":
            from .causal import leave_one_out_leniency
            Z = controls(df).assign(z_exp=df["exp_offset"] * 100, z_len=leave_one_out_leniency(df), v2=df["v2"])
            self._v = sp - sm.OLS(sp, Z).fit().fittedvalues
        self.model = sm.Logit(df["accept"], self._design(df, sp)).fit(disp=0, maxiter=300)
        self._df = df
        return self

    def predict(self, delta_pp: float) -> np.ndarray:
        """Acceptance probability for every training lead if its spread moved by delta_pp."""
        return self.model.predict(self._design(self._df, self._df["spread"] * 100 + delta_pp)).values


class MarginModel:
    """Contribution per bought home (share of OVM value), net of its own spread.

    y = contribution / v_hat − spread  ~  controls + c · exp_offset
    The experiment makes c interpretable: the change in (non-spread) margin of the ACCEPTED pool
    when the spread is exogenously widened — i.e. the winner's-curse slope."""

    def __init__(self, selection_aware: bool = True):
        self.selection_aware = selection_aware

    def fit(self, df: pd.DataFrame) -> "MarginModel":
        b = df[df["accept"] == 1]
        y = b["contribution"] / b["v_hat"] - b["spread"]
        X = controls(b).assign(exp_off_pp=b["exp_offset"] * 100)
        self.fit_ = sm.OLS(y, X).fit(cov_type="HC1")
        # coefficient is per pp of spread; c is dimensionless (share of value per unit of spread)
        self.c = float(self.fit_.params["exp_off_pp"]) * 100 if self.selection_aware else 0.0
        self.c_se = float(self.fit_.bse["exp_off_pp"]) * 100
        self._df = df
        self._g = self.fit_.predict(controls(df).assign(exp_off_pp=0.0)).values
        return self

    def predict(self, delta_pp: float) -> np.ndarray:
        """Expected contribution in $ for every lead if bought at spread + delta."""
        d = delta_pp / 100
        share = self._df["spread"].values + d + self._g + self.c * d
        return share * self._df["v_hat"].values


def true_contribution_if_bought(df: pd.DataFrame, delta_pp: float = 0.0, cfg: SimConfig | None = None) -> np.ndarray:
    """Ground-truth expected contribution ($) per lead if acquired at spread + delta."""
    cfg = cfg or SimConfig()
    r = df["true_list_ratio_policy"].values
    lam = true_dom_scale(df, list_ratio=r, cfg=cfg)
    gap = np.log(r) + np.log(df["v_hat"].values / df["true_value"].values)
    ex = expected_resale(df, r, lam, cfg.weibull_shape, cfg, gap_true=gap)
    offer = df["v_hat"].values * (1 - cfg.fee - df["spread"].values - delta_pp / 100) - df["repair_est"].values
    hold = cfg.holding_cost_per_day * (cfg.reno_days + ex["e_dom"]) * df["v_hat"].values
    return ex["e_sale"] * (1 - cfg.selling_cost) - offer - df["repair_true"].values - hold


def seller_policy_curves(df: pd.DataFrame, grid=np.round(np.arange(-3, 6.01, 0.25), 2), cfg=None) -> dict:
    """Acceptance and expected contribution per lead vs spread shift, for four 'planners'."""
    cfg = cfg or SimConfig()
    acc_cf, acc_nv = AcceptanceModel("cf").fit(df), AcceptanceModel("naive").fit(df)
    mm_sel, mm_nosel = MarginModel(True).fit(df), MarginModel(False).fit(df)
    planners = {
        "truth": lambda d: (true_accept_prob(df, spread=df["spread"].values + d / 100, fee=cfg.fee),
                            true_contribution_if_bought(df, d, cfg)),
        "causal (CF logit + winner's curse)": lambda d: (acc_cf.predict(d), mm_sel.predict(d)),
        "causal acceptance, ignores winner's curse": lambda d: (acc_cf.predict(d), mm_nosel.predict(d)),
        "naive (confounded logit)": lambda d: (acc_nv.predict(d), mm_nosel.predict(d)),
    }
    rows = []
    for name, f in planners.items():
        for d in grid:
            A, m = f(d)
            rows.append(dict(planner=name, delta_pp=d, accept_rate=A.mean(), contribution_per_lead=(A * m).mean(),
                             margin_per_home=(A * m).sum() / A.sum()))
    curves = pd.DataFrame(rows)
    # choose the optimum under each planner, then score that choice under the TRUTH
    truth = curves[curves.planner == "truth"].set_index("delta_pp")
    opt = []
    for name, g in curves.groupby("planner", sort=False):
        d_star = g.loc[g["contribution_per_lead"].idxmax(), "delta_pp"]
        opt.append(dict(planner=name, delta_star_pp=d_star,
                        believed_contribution=g.set_index("delta_pp").loc[d_star, "contribution_per_lead"],
                        true_contribution=truth.loc[d_star, "contribution_per_lead"],
                        true_accept_rate=truth.loc[d_star, "accept_rate"]))
    opt = pd.DataFrame(opt)
    opt["regret_vs_truth_opt"] = opt["true_contribution"].max() - opt["true_contribution"]
    return dict(curves=curves, optimum=opt, winners_curse_slope=mm_sel.c, winners_curse_se=mm_sel.c_se,
                acc_cf=acc_cf, mm_sel=mm_sel)


def volume_constrained_frontier(df: pd.DataFrame, acc: AcceptanceModel, mm: MarginModel,
                                grid=np.round(np.arange(-4, 6.01, 0.25), 2), lambdas=np.linspace(0, 40_000, 81),
                                cfg=None) -> dict:
    """Lagrangian sweep: segment-specific spreads vs one uniform spread."""
    cfg = cfg or SimConfig()
    seg = _segments(df).values
    segs = sorted(set(seg))
    n = len(df)
    # per segment, per delta: predicted and true (sum of contribution, sum of acceptances) per lead
    P = {}
    for d in grid:
        A, m = acc.predict(d), mm.predict(d)
        At = true_accept_prob(df, spread=df["spread"].values + d / 100, fee=cfg.fee)
        mt = true_contribution_if_bought(df, d, cfg)
        for s in segs:
            k = seg == s
            P[(s, d)] = ((A[k] * m[k]).sum() / n, A[k].sum() / n, (At[k] * mt[k]).sum() / n, At[k].sum() / n)
    rows = []
    for lam in lambdas:
        tot = np.zeros(4)
        choice = {}
        for s in segs:
            d_best = max(grid, key=lambda d: P[(s, d)][0] + lam * P[(s, d)][1])
            choice[s] = d_best
            tot += np.array(P[(s, d_best)])
        # uniform delta with the same lambda
        du = max(grid, key=lambda d: sum(P[(s, d)][0] + lam * P[(s, d)][1] for s in segs))
        uni = np.sum([P[(s, du)] for s in segs], axis=0)
        rows.append(dict(shadow_price=lam, seg_pred_contrib=tot[0], seg_pred_volume=tot[1], seg_true_contrib=tot[2],
                         seg_true_volume=tot[3], uni_delta=du, uni_pred_contrib=uni[0], uni_pred_volume=uni[1],
                         uni_true_contrib=uni[2], uni_true_volume=uni[3], **{f"delta_{s}": v for s, v in choice.items()}))
    return dict(frontier=pd.DataFrame(rows), segments=segs)


# =========================================================================== buyer side
class DomModel:
    """Planner's disposition model, estimated from homes we bought.

      DOM ~ Weibull(k, λ_i),  log λ_i = c_i + θ·log r_i        (θ: causal price effect on selling time)
      log sale_i = a_i + η·log r_i + η2·(log r_i)^2            (η: causal pass-through of list to sale price,
                                                                 net of negotiation and stale-listing cuts)
    method='iv' uses the randomized 3% list cut as instrument; 'naive' uses OLS."""

    def __init__(self, method: str = "iv"):
        self.method = method

    def _est(self, y, x, W, b):
        if self.method == "iv":
            r = _iv(y, x, W, b[["list_exp_arm"]])
            coef, se = float(r.params["x"]), float(r.std_errors["x"])
            fitted = (W * r.params[W.columns]).sum(axis=1) + coef * x
        else:
            r = sm.OLS(y, W.assign(x=x)).fit(cov_type="HC1")
            coef, se, fitted = float(r.params["x"]), float(r.bse["x"]), r.fittedvalues
        return coef, se, fitted

    def fit(self, df: pd.DataFrame, cfg: SimConfig | None = None) -> "DomModel":
        self.cfg = cfg or SimConfig()
        b = df[df["accept"] == 1].copy()
        W = controls(b)
        x = np.log(b["list_ratio"]).rename("x")
        ly = np.log(b["dom"])
        self.theta, self.theta_se, fitted = self._est(ly, x, W, b)
        resid = ly - fitted
        self.k = float(weibull_min.fit(np.exp(resid - resid.mean()), floc=0)[0])
        self.c = (fitted - self.theta * x + EULER / self.k).values       # E[log DOM] = log λ − γ/k
        ls = np.log(b["sale_price"])
        self.eta, self.eta_se, fs = self._est(ls, x, W, b)
        self.a = (fs - self.eta * x).values + np.log(np.mean(np.exp(ls - fs)))   # smearing
        # curvature: with three randomized arms (-3%, 0, +3%) we can identify a quadratic
        # price->sale-price curve (over-pricing gets negotiated away, so pass-through falls as price rises)
        self.eta2 = 0.0
        if self.method == "iv":
            Z = pd.DataFrame({"arm_m": (b["list_exp_arm"] == -1).astype(float),
                              "arm_p": (b["list_exp_arm"] == 1).astype(float)}, index=b.index)
            X2 = pd.DataFrame({"x": x, "x2": x ** 2})
            r = IV2SLS(ls, W, X2, Z).fit(cov_type="robust")
            self.eta, self.eta2 = float(r.params["x"]), float(r.params["x2"])
            self.eta_se = float(r.std_errors["x"])
            fs = (W * r.params[W.columns]).sum(axis=1) + self.eta * x + self.eta2 * x ** 2
            self.a = (fs - self.eta * x - self.eta2 * x ** 2).values + np.log(np.mean(np.exp(ls - fs)))
        self.b = b
        return self

    def outcomes(self, rho: float, theta: float | None = None, eta: float | None = None) -> dict:
        th = self.theta if theta is None else theta
        et = self.eta if eta is None else eta
        lr = np.log(self.b["list_ratio"].values * rho)
        lam = np.exp(self.c + th * lr)
        e_dom = lam * G(1 + 1 / self.k)
        e_sale = np.exp(self.a + et * lr + self.eta2 * lr ** 2)
        return _pack(self.b, rho, e_sale, e_dom, 1 - np.exp(-(60 / lam) ** self.k), self.cfg)


def _pack(b, rho, e_sale, e_dom, p60, cfg) -> dict:
    hold = cfg.holding_cost_per_day * (cfg.reno_days + e_dom) * b["v_hat"].values
    contrib = e_sale * (1 - cfg.selling_cost) - hold - b["offer"].values - b["repair_true"].values
    return dict(rho=rho, p_sold_60d=p60.mean(), exp_dom=e_dom.mean(), exp_sale=e_sale.mean(),
                holding=hold.mean(), contribution=contrib.mean(), contribution_margin=contrib.sum() / e_sale.sum())


def true_buyer_outcomes(b: pd.DataFrame, rho: float, cfg: SimConfig) -> dict:
    r = b["list_ratio"].values * rho
    lam = true_dom_scale(b, list_ratio=r, cfg=cfg)
    gap = np.log(r) + np.log(b["v_hat"].values / b["true_value"].values)
    ex = expected_resale(b, r, lam, cfg.weibull_shape, cfg, gap_true=gap)
    return _pack(b, rho, ex["e_sale"], ex["e_dom"], ex["p60"], cfg)


def buyer_policy_curves(df: pd.DataFrame, grid=np.round(np.arange(0.92, 1.0801, 0.005), 3), cfg=None) -> dict:
    cfg = cfg or SimConfig()
    iv, nv = DomModel("iv").fit(df, cfg), DomModel("naive").fit(df, cfg)
    b = iv.b
    rows = []
    for rho in grid:
        rows.append(dict(planner="truth", **true_buyer_outcomes(b, rho, cfg)))
        rows.append(dict(planner="causal (2SLS)", **iv.outcomes(rho)))
        rows.append(dict(planner="naive (OLS)", **nv.outcomes(rho)))
    curves = pd.DataFrame(rows)
    truth = curves[curves.planner == "truth"].set_index("rho")
    opt = []
    for name, g in curves.groupby("planner", sort=False):
        r_star = g.loc[g["contribution"].idxmax(), "rho"]
        opt.append(dict(planner=name, rho_star=r_star, believed_contribution=g.set_index("rho").loc[r_star, "contribution"],
                        true_contribution=truth.loc[r_star, "contribution"],
                        true_p_sold_60d=truth.loc[r_star, "p_sold_60d"]))
    opt = pd.DataFrame(opt)
    opt["regret_vs_truth_opt"] = opt["true_contribution"].max() - opt["true_contribution"]
    return dict(curves=curves, optimum=opt, iv=iv, naive=nv)


def three_percent_cut_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Worked example: list at the current price vs 3% lower."""
    c = curves[curves["rho"].isin([1.0, 0.97])].copy()
    wide = c.pivot(index="planner", columns="rho",
                   values=["p_sold_60d", "exp_dom", "exp_sale", "holding", "contribution"])
    out = pd.DataFrame({
        "P(sold<=60d) now": wide[("p_sold_60d", 1.0)],
        "P(sold<=60d) at -3%": wide[("p_sold_60d", 0.97)],
        "Δ P60 (pp)": (wide[("p_sold_60d", 0.97)] - wide[("p_sold_60d", 1.0)]) * 100,
        "E[DOM] now": wide[("exp_dom", 1.0)],
        "E[DOM] at -3%": wide[("exp_dom", 0.97)],
        "Δ revenue ($)": wide[("exp_sale", 0.97)] - wide[("exp_sale", 1.0)],
        "Δ holding cost ($)": wide[("holding", 0.97)] - wide[("holding", 1.0)],
        "Δ contribution ($/home)": wide[("contribution", 0.97)] - wide[("contribution", 1.0)],
    })
    return out


def theta_uncertainty(dm: DomModel, n_draws: int = 80, grid=np.round(np.arange(0.92, 1.0801, 0.0025), 4),
                      seed: int = 0) -> dict:
    """Propagate sampling uncertainty in (θ, η) into the optimal list price."""
    rng = np.random.default_rng(seed)
    th = rng.normal(dm.theta, dm.theta_se, n_draws)
    et = rng.normal(dm.eta, dm.eta_se, n_draws)
    prof = np.array([[dm.outcomes(r, t, e)["contribution"] for r in grid] for t, e in zip(th, et)])
    rstar = grid[prof.argmax(axis=1)]
    robust = grid[prof.mean(axis=0).argmax()]
    return dict(theta_draws=th, eta_draws=et, rho_star_draws=rstar, rho_robust=robust, grid=grid, profit_matrix=prof)

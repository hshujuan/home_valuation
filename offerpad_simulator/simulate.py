"""Simulate an iBuyer funnel with KNOWN causal structure.

Causal graph (seller side)

    X (observed features) ──► V (true value) ──► R (seller's self-appreciated reservation value)
    U (condition, unobserved in data) ──► V, R  and ──► analyst adjustment ──► spread
    Z_exp  (randomized spread arm)      ──► spread             [valid IV]
    Z_jud  (analyst leniency)           ──► spread             [valid IV if assignment is quasi-random]
    v2     (algorithm version cutover)  ──► spread             [valid IV / RD-in-time with time controls]
    rate   (mortgage rate deviation)    ──► spread AND ──► R   [INVALID IV: exclusion violated]
    spread ──► offer ──► accept = 1[offer beats seller's net reservation, noisy]

Buyer side (homes we bought)

    U, local demand heat D ──► list ratio (pricing team sees both)  and ──► days on market
    Z_list (randomized list -3% / 0 / +3%) ──► list ratio                   [valid IV]
    list / true value ──► DOM (Weibull, TRUE elasticity theta) ──► sold within 60 days, contribution

Everything the estimators need to be checked against is stored in columns
prefixed `true_` or in the helper functions `true_accept_prob` / `true_dom_scale`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimConfig
from .valuation import OVM

URGENCY = np.array(["low", "medium", "high"])


# --------------------------------------------------------------------------- homes
def _homes(n: int, cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    market = rng.integers(0, cfg.n_markets, n)
    beds = rng.choice([2, 3, 4, 5], n, p=[0.15, 0.45, 0.30, 0.10])
    sqft = np.clip(rng.normal(700 + 380 * beds, 260, n), 650, 5500)
    baths = np.clip(np.round(rng.normal(0.6 * beds + 0.4, 0.45, n) * 2) / 2, 1, 5)
    age = np.clip(rng.gamma(2.2, 11, n), 0, 90)
    lot = np.clip(rng.lognormal(np.log(0.18), 0.5, n), 0.03, 2.0)
    school = np.clip(rng.normal(6, 1.8, n), 1, 10)
    month = rng.integers(1, 13, n)
    U = rng.normal(0, 1, n)                                    # condition / curb appeal / view
    idio_sd = 0.015 + 0.0006 * age                             # old homes are harder to value
    log_v = (12.55 + 0.62 * np.log(sqft / 1800) + 0.025 * (beds - 3) + 0.035 * (baths - 2)
             - 0.0025 * (age - 25) + 0.08 * np.log(lot / 0.18) + 0.025 * (school - 6)
             + np.asarray(cfg.market_effects)[market]
             + 0.05 * (age > 55)                                    # vintage premium (nonlinear)
             + 0.02 * (school - 6) * (market >= 3)                  # schools matter more in pricier markets
             + cfg.beta_condition * U * (1 + age / 60)          # condition matters more for old homes
             + rng.normal(0, idio_sd))
    return pd.DataFrame(dict(market=market, beds=beds, sqft=sqft.round(), baths=baths, age=age.round(1),
                             lot_acres=lot.round(3), school_score=school.round(1), month=month,
                             true_U=U, true_value=np.exp(log_v)))


def simulate_history(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Closed MLS sales used to train the OVM (price = value * small transaction noise)."""
    h = _homes(cfg.n_history, cfg, rng)
    h["sale_price"] = h["true_value"] * np.exp(rng.normal(0, 0.02, len(h)))
    return h


# --------------------------------------------------------------------------- truth helpers
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def offer_amount(v_hat, spread, fee, repair_est):
    return v_hat * (1.0 - fee - spread) - repair_est


def true_accept_prob(df: pd.DataFrame, spread=None, price_mult: float = 1.0, fee: float = 0.05) -> np.ndarray:
    """Ground-truth P(accept) for each lead at an arbitrary spread (counterfactual).

    `price_mult` scales the whole offer (e.g. 0.97 = a 3% lower offer)."""
    s = df["spread"].values if spread is None else np.broadcast_to(spread, len(df))
    offer = offer_amount(df["v_hat"].values, s, fee, df["repair_est"].values) * price_mult
    gap = np.log(np.maximum(offer, 1.0)) - df["true_log_reservation_net"].values
    return _sigmoid(df["true_accept_intercept"].values + df["true_k"].values * gap)


def true_dom_scale(df: pd.DataFrame, list_ratio=None, cfg: SimConfig | None = None) -> np.ndarray:
    cfg = cfg or SimConfig()
    r = df["list_ratio"].values if list_ratio is None else np.broadcast_to(list_ratio, len(df))
    gap = np.log(r) + np.log(df["v_hat"].values / df["true_value"].values)   # list vs TRUE value
    return np.exp(np.log(cfg.dom_scale_days) + cfg.theta_price * gap - cfg.demand_on_dom * df["true_demand"].values)


def sale_price_multiplier(gap_true, dom, cfg: SimConfig) -> np.ndarray:
    """sale / list = (1 - concession) x negotiation give-back on over-pricing x stale-listing decay."""
    give = np.exp(-cfg.overprice_giveback * np.maximum(gap_true, 0.0))
    decay = np.exp(-np.minimum(cfg.decay_per_day * np.maximum(dom - cfg.decay_start_day, 0.0), cfg.max_decay))
    return (1 - cfg.concession) * give * decay


_U = (np.arange(64) + 0.5) / 64          # quantile grid for Weibull expectations


def expected_resale(df, list_ratio, lam, k, cfg: SimConfig, gap_true=None) -> dict:
    """E[sale price], E[DOM], P(sold<=60d) by quadrature over the Weibull DOM distribution."""
    dom = lam[:, None] * (-np.log(1 - _U[None, :])) ** (1 / k)
    g = 0.0 if gap_true is None else gap_true[:, None]
    mult = sale_price_multiplier(g, dom, cfg).mean(axis=1)
    lst = df["v_hat_resale"].values * list_ratio
    return dict(e_sale=lst * mult, e_dom=dom.mean(axis=1), p60=1 - np.exp(-(60 / lam) ** k))


def true_p_sold_within(df, days=60, list_ratio=None, cfg=None) -> np.ndarray:
    cfg = cfg or SimConfig()
    lam = true_dom_scale(df, list_ratio, cfg)
    return 1.0 - np.exp(-(days / lam) ** cfg.weibull_shape)


# --------------------------------------------------------------------------- funnel
def simulate(cfg: SimConfig | None = None, return_ovm: bool = False):
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)

    hist = simulate_history(cfg, rng)
    ovm = OVM(seed=cfg.seed).fit(hist)

    n = cfg.n_leads
    df = _homes(n, cfg, rng)
    df["day"] = rng.integers(0, cfg.n_days, n)
    df["month"] = (df["day"] // 30) % 12 + 1
    df = pd.concat([df, ovm.predict(df)], axis=1)

    # ---- macro series: mortgage-rate deviation (pp/100) and market-month demand heat
    rate_path = np.cumsum(rng.normal(0, 0.0004, cfg.n_days))
    rate_path = rate_path - rate_path.mean()
    df["rate_dev"] = rate_path[df["day"].values]
    heat = rng.normal(0, 1, (cfg.n_markets, cfg.n_days // 30 + 2))
    df["true_demand"] = heat[df["market"].values, df["day"].values // 30]

    # ---- seller type (collected in the offer funnel -> observed)
    df["urgency"] = rng.choice(URGENCY, n, p=[0.45, 0.35, 0.20])
    df["has_agent_quote"] = rng.binomial(1, 0.35, n)

    # ---- repairs: inspection-based estimate deducted from the offer
    repair_true = np.maximum(1000, 2500 + 160 * df["age"] - 3500 * df["true_U"] + rng.normal(0, 1500, n))
    df["repair_true"] = repair_true
    df["repair_est"] = (repair_true * np.exp(rng.normal(0, cfg.repair_overrun_sd, n))).round(-2)

    # ---- spread policy: algorithm + analyst + experiment + v2 + rates
    df["analyst_id"] = rng.integers(0, cfg.n_analysts, n)          # queue-based, quasi-random
    leniency = rng.normal(0, cfg.leniency_sd, cfg.n_analysts)
    u_soft = df["true_U"] + rng.normal(0, cfg.analyst_soft_noise, n)
    in_exp = rng.random(n) < cfg.experiment_share
    exp_off = np.where(in_exp, rng.choice(cfg.experiment_offsets, n), 0.0)
    df["in_experiment"] = in_exp.astype(int)
    df["exp_offset"] = exp_off
    df["v2"] = (df["day"] >= cfg.v2_day).astype(int)

    spread_algo = cfg.spread_base + cfg.spread_per_uncertainty * df["unc_halfwidth"] + 0.003 * (df["market"] % 3)
    df["spread_algo"] = spread_algo
    spread = (spread_algo
              - cfg.analyst_soft_info * u_soft                      # CONFOUNDER path (analyst sees condition)
              + leniency[df["analyst_id"].values]                   # IV: analyst leniency
              + exp_off                                             # IV: randomized test arm
              + cfg.v2_shift * df["v2"]                             # IV: algorithm version
              + cfg.rate_passthrough * df["rate_dev"]               # "IV" that violates exclusion
              + rng.normal(0, 0.003, n))
    df["spread"] = np.clip(spread, 0.0, 0.18)
    df["offer"] = offer_amount(df["v_hat"], df["spread"], cfg.fee, df["repair_est"])

    # ---- seller's self-appreciated value and net reservation price
    urg = df["urgency"].values
    past_hpa = np.asarray(cfg.past_hpa_12m)[df["market"].values]
    df["past_hpa_12m"] = past_hpa
    self_bias = (cfg.self_apprec_base + cfg.self_apprec_hpa_passthrough * past_hpa
                 + cfg.rate_on_reservation * df["rate_dev"] + rng.normal(0, cfg.self_apprec_sd, n))
    df["true_self_apprec"] = self_bias                                # log(seller belief / true value)
    df["true_seller_belief"] = df["true_value"] * np.exp(self_bias)
    conv = np.vectorize(cfg.convenience_by_urgency.get)(urg)
    trad = np.clip(rng.normal(cfg.traditional_sale_cost, 0.015, n), 0.03, 0.14)
    df["true_log_reservation_net"] = np.log(df["true_seller_belief"] * (1 - trad - conv))
    k = np.vectorize(cfg.k_by_urgency.get)(urg) * np.where(df["has_agent_quote"] == 1, cfg.k_quote_multiplier, 1.0)
    df["true_k"] = k
    df["true_accept_intercept"] = cfg.accept_intercept + 0.25 * np.sin(2 * np.pi * df["day"] / 365)
    df["true_p_accept"] = true_accept_prob(df, fee=cfg.fee)
    df["accept"] = rng.binomial(1, df["true_p_accept"])

    # ---- resale for acquired homes
    fwd_hpa = 0.4 * past_hpa + rng.normal(0, 0.01, cfg.n_markets)[df["market"].values]
    growth = np.exp(fwd_hpa * cfg.reno_days / 365)
    df["v_hat_resale"] = df["v_hat"] * growth
    df["true_value_at_list"] = df["true_value"] * growth
    u_soft2 = df["true_U"] + rng.normal(0, 0.4, n)
    in_lexp = rng.random(n) < cfg.list_experiment_share
    arm = np.where(in_lexp, rng.integers(0, len(cfg.list_experiment_arms), n), 1)
    df["list_exp_offset"] = np.asarray(cfg.list_experiment_arms)[arm]           # -3% / 0 / +3%
    df["list_exp_arm"] = np.sign(df["list_exp_offset"])                          # -1 / 0 / +1
    df["in_list_experiment"] = in_lexp.astype(int)
    log_r = (cfg.list_soft_info * u_soft2 + cfg.list_demand_response * df["true_demand"]
             + np.log1p(df["list_exp_offset"]) + rng.normal(0, 0.01, n))
    df["list_ratio"] = np.exp(log_r)                                  # list price / OVM resale value
    df["list_price"] = df["v_hat_resale"] * df["list_ratio"]
    lam = true_dom_scale(df, cfg=cfg)
    df["dom"] = np.ceil(lam * rng.weibull(cfg.weibull_shape, n))
    df["sold_60d"] = (df["dom"] <= 60).astype(int)
    df["true_p_sold_60d"] = true_p_sold_within(df, 60, cfg=cfg)
    gap_true = np.log(df["list_price"] / df["true_value_at_list"])
    df["sale_price"] = df["list_price"] * sale_price_multiplier(gap_true, df["dom"], cfg)
    df["holding_cost"] = cfg.holding_cost_per_day * (cfg.reno_days + df["dom"]) * df["v_hat"]
    df["gross_profit"] = df["sale_price"] - df["offer"] - df["repair_true"]
    df["contribution"] = df["gross_profit"] - cfg.selling_cost * df["sale_price"] - df["holding_cost"]

    # keep the would-be list ratio for EVERY lead (ground truth for counterfactual planning)
    df["true_list_ratio_policy"] = df["list_ratio"]
    # seller's answer to "what do you think your home is worth?" (asked in the funnel, 60% answer)
    answered = rng.random(n) < 0.6
    df["seller_stated_value"] = np.where(answered, df["true_seller_belief"] * np.exp(rng.normal(0, 0.02, n)), np.nan)

    # resale columns only exist for homes we actually bought
    resale_cols = ["list_ratio", "list_price", "dom", "sold_60d", "true_p_sold_60d", "sale_price",
                   "holding_cost", "gross_profit", "contribution", "list_exp_arm", "list_exp_offset",
                   "in_list_experiment"]
    df.loc[df["accept"] == 0, resale_cols] = np.nan

    df.insert(0, "lead_id", np.arange(n))
    return (df, ovm, hist) if return_ovm else df


def observed_columns(df: pd.DataFrame) -> list[str]:
    """Columns an analyst at the company would actually see (no `true_` columns)."""
    return [c for c in df.columns if not c.startswith("true_")]


if __name__ == "__main__":
    d = simulate()
    print(d[observed_columns(d)].describe().T.round(3))
    print("acceptance rate", d["accept"].mean().round(3))

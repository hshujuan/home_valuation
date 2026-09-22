"""Structural generators. Returned truth tables must never be model features."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax

from .features import add_comps

ACTIONS = np.array([-0.06, -0.03, 0.0, 0.03])
MARKETS = np.array(["Metro-A", "Metro-B", "Metro-C", "Metro-D"])


@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_market: int = 5000
    n_sellers: int = 4000
    n_resale: int = 6000
    n_iv: int = 8000

    def __post_init__(self):
        if min(self.n_market, self.n_sellers, self.n_resale, self.n_iv) < 300:
            raise ValueError("Use at least 300 observations in each simulation table.")

    def to_dict(self):
        return asdict(self)


def properties(n, rng, start_day=0, end_day=730):
    frame = pd.DataFrame({
        "home_id": np.arange(n),
        "market": rng.choice(MARKETS, n),
        "sqft": np.round(rng.lognormal(np.log(1800), 0.28, n)),
        "age": rng.integers(0, 75, n),
        "condition": rng.uniform(0.15, 1, n),
        "market_day": rng.integers(start_day, end_day, n),
    })
    frame["beds"] = np.maximum(1, np.round(frame["sqft"] / 650)).astype(int)
    offsets = frame["market"].map(dict(zip(MARKETS, [-0.1, 0.08, 0.2, -0.02])))
    log_value = (
        np.log(460_000) + 0.65 * np.log(frame["sqft"] / 1800)
        - 0.0025 * frame["age"] + 0.12 * frame["condition"] ** 2
        + offsets + 0.00010 * frame["market_day"] + 0.025 * np.sin(frame["market_day"] / 90)
    )
    quality = rng.normal(0, 0.055, n)
    true_value = np.exp(log_value + quality)
    frame["reported_condition"] = np.clip(frame.condition + rng.normal(0.1, 0.12, n), 0, 1)
    frame["condition"] = np.clip(frame.condition + rng.normal(0, 0.025, n), 0, 1)
    return frame, np.asarray(true_value), quality


def market_sales(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    frame, value, quality = properties(n, rng)
    frame["sale_price"] = value * np.exp(rng.normal(-0.025**2 / 2, 0.025, n))
    frame.loc[rng.random(n) < 0.06, "condition"] = np.nan
    truth = pd.DataFrame({"home_id": frame.home_id, "true_value": value,
                          "private_quality": quality})
    return frame.sort_values("market_day").reset_index(drop=True), truth


def resale_population(n, seed, history, avm):
    rng = np.random.default_rng(seed)
    frame, value, quality = properties(n * 3, rng, 760, 1120)
    frame["avm_value"] = avm.predict(add_comps(frame, history))
    frame["base_list_price"] = frame["avm_value"] * np.exp(rng.normal(0.035, 0.035, len(frame)))
    frame["days_on_market"] = rng.integers(14, 65, len(frame))
    demand = rng.normal(0, 1, len(frame))
    premium = np.log(frame["base_list_price"] / value)
    frame["views_before"] = rng.poisson(np.exp(4.0 + 0.3 * demand - 2 * premium))
    still_unsold = rng.random(len(frame)) < expit(
        0.3 + 6 * premium - 0.7 * demand + 0.006 * frame["days_on_market"]
    )
    selected = np.flatnonzero(still_unsold)[:n]
    if len(selected) != n:
        raise RuntimeError("Insufficient eligible homes; increase the candidate population.")
    frame = frame.iloc[selected].reset_index(drop=True)
    value, quality, demand = value[selected], quality[selected], demand[selected]
    frame["demand_signal"] = 0.45 * demand + rng.normal(0, 0.8, n)
    frame["cost_basis"] = frame["avm_value"] * 0.86
    frame["holding_per_day"] = frame["avm_value"] * 0.00010
    frame["decision_date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(frame.market_day, "D")
    frame["listing_date"] = frame["decision_date"] - pd.to_timedelta(frame.days_on_market, "D")
    frame["condition_verified_at"] = frame["listing_date"] - pd.Timedelta(days=1)
    truth = pd.DataFrame({
        "home_id": frame.home_id, "true_value": value, "private_quality": quality,
        "hidden_demand": demand, "completion_uniform": rng.random(n),
    })
    return frame, truth


def oracle_resale_q(frame, truth, action):
    price = frame["base_list_price"].to_numpy() * (1 + np.asarray(action))
    sensitivity = 12 + 8 * frame["condition"].to_numpy()
    z = (
        -0.1 + 0.35 * np.log(frame["sqft"].to_numpy() / 1800)
        + 0.3 * frame["condition"].to_numpy()
        - 0.004 * frame["days_on_market"].to_numpy()
        + 0.85 * truth["hidden_demand"].to_numpy()
        - sensitivity * np.log(price / truth["true_value"].to_numpy())
    )
    return expit(z)


def resale_log(population, truth, seed, design="randomized"):
    """Assign actions only after eligibility. Future liquidation is a fixed study policy."""
    rng = np.random.default_rng(seed)
    frame, oracle = population.copy(), truth.copy()
    n = len(frame)
    if design == "multiarm":
        action = rng.choice(ACTIONS, n)
        probability = np.full(n, 1 / len(ACTIONS))
    elif design == "confounded_multiarm":
        weights = softmax(-ACTIONS[None, :] * (
            10 - 35 * truth.hidden_demand.to_numpy()[:, None]), axis=1)
        indices = (rng.random(n)[:, None] > weights.cumsum(axis=1)).sum(axis=1)
        action = ACTIONS[indices]
        probability = weights[np.arange(n), indices]
    elif design == "randomized":
        probability = np.full(n, 0.5)
        action = -0.03 * rng.binomial(1, probability)
    elif design in {"observed", "hidden"}:
        if design == "observed":
            frame["demand_signal"] = truth["hidden_demand"]
        probability = expit(
            -0.2 + 0.025 * (frame["days_on_market"] - 35)
            - 1.5 * truth["hidden_demand"]
        ).to_numpy()
        action = -0.03 * rng.binomial(1, probability)
    else:
        raise ValueError(f"Unknown assignment design: {design}")
    frame["design"] = design
    frame["action"] = action
    frame["assigned_action"] = action
    frame["override_reason"] = "none; full compliance in this price experiment"
    frame["cut"] = np.isclose(action, -0.03).astype(int)
    frame["list_price"] = frame["base_list_price"] * (1 + action)
    frame["assignment_probability"] = probability if design in {"multiarm", "randomized"} else np.nan
    q = oracle_resale_q(frame, truth, action)
    sold = truth["completion_uniform"].to_numpy() < q
    frame["sold_30d"] = sold.astype(int)
    frame["label_matured"] = True
    frame["negotiation_discount"] = 0.01
    sale = frame["list_price"].to_numpy() * 0.99 * np.exp(rng.normal(-0.008**2 / 2, 0.008, n))
    terminal = truth["true_value"].to_numpy() * 0.94 * np.exp(rng.normal(-0.02**2 / 2, 0.02, n))
    close_days = np.where(sold, rng.integers(10, 27, n), 45)
    contract_days = np.where(sold, np.maximum(1, close_days - 7), 35)
    frame["contract_date"] = frame["decision_date"] + pd.to_timedelta(contract_days, "D")
    frame["close_date"] = frame["decision_date"] + pd.to_timedelta(close_days, "D")
    frame["observed_through"] = frame["decision_date"] + pd.Timedelta(days=60)
    frame["contract_cancelled_before_terminal"] = (~sold) & (rng.random(n) < 0.18)
    frame["sale_price_30d"] = np.where(sold, sale, np.nan)
    frame["terminal_sale_price"] = np.where(sold, np.nan, terminal)
    frame["realized_revenue"] = np.where(sold, sale, terminal)
    frame["realized_gross_profit"] = frame.realized_revenue - frame.cost_basis
    frame["realized_contribution"] = (
        frame.realized_gross_profit - 0.02 * frame.realized_revenue
        - frame.holding_per_day * close_days
        - frame.cost_basis * 0.08 / 365 * close_days
    )
    oracle["true_assignment_probability"] = probability
    oracle["q_hold"] = oracle_resale_q(frame, truth, 0)
    oracle["q_cut3"] = oracle_resale_q(frame, truth, -0.03)
    return frame, oracle


def seller_offers(n, seed, history, avm):
    rng = np.random.default_rng(seed)
    frame, value, quality = properties(n, rng, 760, 1000)
    frame["avm_value"] = avm.predict(add_comps(frame, history))
    attachment = rng.uniform(0, 0.07, n)
    convenience = rng.uniform(0.01, 0.09, n)
    belief = value * np.exp(rng.normal(0.02, 0.025, n))
    reservation = belief * (0.98 + attachment - convenience)
    frame["belief_before_offer"] = belief * np.exp(rng.normal(0, 0.01, n))
    frame["attachment_score"] = attachment
    frame["convenience_score"] = convenience
    frame["service_rate"] = 0.02
    frame["repair_credit"] = 6000.0
    frame["seller_closing_cost"] = 3000.0
    frame["offer_ratio"] = rng.choice([0.90, 0.94, 0.98, 1.02], n)
    frame["offer"] = frame["avm_value"] * frame["offer_ratio"]
    frame["offer_date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(frame.market_day, "D")
    frame["condition_verified_at"] = frame.offer_date - pd.Timedelta(days=1)
    frame["belief_elicited_at"] = frame.offer_date - pd.Timedelta(days=2)
    frame["company_outlay"] = frame.offer * (1 - frame.service_rate) - frame.repair_credit
    frame["net_proceeds"] = frame.company_outlay - frame.seller_closing_cost
    probability = expit((frame.net_proceeds - reservation) / (0.025 * frame.avm_value))
    frame["accepted_14d"] = rng.binomial(1, probability)
    frame["completed_acquisition"] = frame.accepted_14d * rng.binomial(1, 0.96, n)
    frame["offer_assignment_probability"] = 0.25
    truth = pd.DataFrame({
        "home_id": frame.home_id, "true_value": value, "private_quality": quality,
        "belief": belief, "reservation_value": reservation,
        "acceptance_probability": probability,
    })
    return frame, truth


def iv_sample(n=8000, seed=42, scenario="strong"):
    settings = {"strong": (2.1, 0, False), "weak": (0.07, 0, False),
                "invalid": (2.1, 0.65, False), "nonrandom": (2.1, 0, True)}
    if scenario not in settings:
        raise ValueError(f"Unknown IV scenario: {scenario}")
    strength, direct, nonrandom = settings[scenario]
    rng = np.random.default_rng(seed)
    x, u = rng.normal(size=(2, n))
    z = rng.binomial(1, expit(1.4 * u) if nonrandom else np.full(n, 0.5))
    compliance_draw = rng.random(n)
    d0 = compliance_draw < expit(-0.7 + 0.5 * x + 0.9 * u)
    d1 = compliance_draw < expit(-0.7 + 0.5 * x + 0.9 * u + strength)
    d = np.where(z == 1, d1, d0).astype(int)
    base = -0.6 + 0.4 * x + 0.9 * u
    effect = 0.6 + 0.35 * expit(x)
    q0, q1 = expit(base), expit(base + effect)
    y = rng.binomial(1, expit(base + effect * d + direct * z))
    compliers = d1 & ~d0
    truth = {
        "oracle_ate": float(np.mean(q1 - q0)),
        "oracle_late": float(np.mean((q1 - q0)[compliers])) if compliers.any() else None,
        "complier_fraction": float(compliers.mean()),
        "exclusion_holds": direct == 0, "independence_holds": not nonrandom,
    }
    return pd.DataFrame({"x": x, "recommendation": z, "cut": d, "sold_30d": y}), truth


def reviewer_sample(n=6000, seed=42, direct_effect=False):
    rng = np.random.default_rng(seed)
    propensity = np.linspace(-1.2, 1.2, 16)
    history_ids = rng.integers(0, 16, 5000)
    historical_cut = rng.binomial(1, expit(propensity[history_ids]))
    history = pd.DataFrame({"reviewer": history_ids, "cut": historical_cut})
    scores = history.groupby("reviewer")["cut"].mean()
    reviewer = rng.integers(0, 16, n)
    x, u = rng.normal(size=(2, n))
    cut = rng.binomial(1, expit(propensity[reviewer] + 0.4 * x + 0.7 * u))
    direct = 0.7 * propensity[reviewer] if direct_effect else 0
    y = rng.binomial(1, expit(-0.5 + 0.4 * x + 0.8 * u + 0.8 * cut + direct))
    return pd.DataFrame({"x": x, "reviewer": reviewer, "cut": cut, "sold_30d": y,
                         "recommendation": pd.Series(reviewer).map(scores).to_numpy()})


def toy_cut_probability(q0, log_odds_slope, cut=0.03):
    if not 0 < q0 < 1 or not 0 <= cut < 1:
        raise ValueError("q0 must be inside (0, 1); cut must be in [0, 1).")
    return float(expit(logit(q0) + log_odds_slope * np.log1p(-cut)))

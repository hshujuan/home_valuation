"""A separate linked experiment; observed decisions never contain simulator truth."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from .features import add_comps
from .seller import offer_ledger
from .simulation import properties


STAGE_ACTIONS = np.array([0.0, 0.01, 0.02, 0.03])
QUOTE_COST = 200.0
SELLING_RATE = 0.02


@dataclass(frozen=True)
class TwoSidedConfig:
    seed: int = 42
    n_market: int = 5000
    n_development: int = 12000
    n_continuation: int = 10000
    n_evaluation: int = 8000

    def __post_init__(self):
        if self.n_market < 500 or min(
                self.n_development, self.n_continuation, self.n_evaluation) < 2400:
            raise ValueError("Use at least 500 market sales and 2400 prospects per linked cohort.")

    def to_dict(self):
        return asdict(self)


def feasible_actions(frame, stage):
    if stage == "buy":
        cap = frame.max_escalation.to_numpy()
        return STAGE_ACTIONS[None, :] <= cap[:, None] + 1e-12
    if stage == "sell":
        prices = frame.base_list_price.to_numpy()[:, None] * (1 - STAGE_ACTIONS)
        return prices >= frame.minimum_list_price.to_numpy()[:, None]
    raise ValueError("Stage must be buy or sell.")


def action_indices(actions):
    matches = np.isclose(np.asarray(actions)[:, None], STAGE_ACTIONS)
    if not np.all(matches.sum(axis=1) == 1):
        raise ValueError("Unsupported action: use magnitudes 0%, 1%, 2%, or 3%.")
    return matches.argmax(axis=1)


def randomize(feasible, rng):
    if not feasible.any(axis=1).all():
        raise ValueError("No feasible randomized action for at least one prospect.")
    counts = feasible.sum(axis=1)
    rank = (rng.random(len(feasible)) * counts).astype(int)
    indices = (feasible.cumsum(axis=1) > rank[:, None]).argmax(axis=1)
    return STAGE_ACTIONS[indices], 1 / counts


def generate_cohort(n, seed, name, start_day, history, avm):
    rng = np.random.default_rng(seed)
    frame, true_value, quality = properties(n, rng, start_day, start_day + 220)
    frame["home_id"] = [f"{name}:{i}" for i in range(n)]
    frame["cohort"] = name
    frame["avm_value"] = avm.predict(add_comps(frame, history))
    frame["belief_before_offer"] = true_value * np.exp(rng.normal(0.02, 0.03, n))
    frame["attachment_score"] = rng.uniform(0, 0.07, n)
    frame["convenience_score"] = rng.uniform(0.01, 0.09, n)
    reservation = frame.belief_before_offer.to_numpy() * (
        0.98 + frame.attachment_score.to_numpy() - frame.convenience_score.to_numpy())
    reservation *= np.exp(rng.normal(0, 0.018, n))
    frame["service_rate"] = 0.02
    frame["repair_credit"] = 6000.0
    frame["seller_closing_cost"] = 3000.0
    frame["initial_offer_ratio"] = rng.choice([0.86, 0.90, 0.94], n)
    frame = offer_ledger(frame, frame.initial_offer_ratio)
    frame["initial_offer"] = frame.offer
    frame["initial_outlay"] = frame.company_outlay
    frame["repair_forecast"] = 8000 + 12000 * (1 - frame.condition) + 80 * frame.age
    frame["cash_holding_per_day"] = 20 + 0.00002 * frame.avm_value
    forecast_daily = frame.cash_holding_per_day + 0.08 * frame.initial_outlay / 365
    frame["margin_forecast"] = (
        0.98 * (1 - SELLING_RATE) * frame.avm_value - frame.repair_forecast
        - frame.initial_outlay - 70 * forecast_daily
    )
    extra_outlay = frame.initial_offer.to_numpy()[:, None] * 0.98 * STAGE_ACTIONS
    permitted = frame.margin_forecast.to_numpy()[:, None] - extra_outlay >= 5000
    permitted[:, 0] = True
    frame["max_escalation"] = np.where(permitted, STAGE_ACTIONS, 0).max(axis=1)
    frame["offer_date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(frame.market_day, "D")
    frame["buy_decision_date"] = frame.offer_date + pd.Timedelta(days=7)
    frame["buy_features_at"] = frame.buy_decision_date - pd.Timedelta(days=1)
    frame["buy_mature_at"] = frame.buy_decision_date + pd.Timedelta(days=35)
    frame["reward_mature_at"] = frame.offer_date + pd.Timedelta(days=150)
    frame["observed_through"] = frame.reward_mature_at
    initial_probability = expit(
        (frame.net_proceeds.to_numpy() - reservation) / (0.035 * frame.avm_value.to_numpy()))
    initial = rng.random(n) < initial_probability
    self_gap = (frame.belief_before_offer - frame.initial_offer) / frame.avm_value
    frame["engagement_score"] = expit(0.6 - 2 * self_gap + rng.normal(0, 0.5, n))
    engaged = rng.random(n) < frame.engagement_score
    eligible = ~initial & engaged
    frame["initial_accepted_7d"] = initial.astype(int)
    frame["buy_eligible"] = eligible
    action, probability = randomize(feasible_actions(frame, "buy"), rng)
    frame["buy_action"] = np.where(eligible, action, 0)
    frame["buy_assignment_probability"] = np.where(eligible, probability, np.nan)
    final = offer_ledger(frame, frame.initial_offer_ratio * (1 + frame.buy_action))
    frame["company_outlay"] = final.company_outlay
    frame["final_offer"] = final.offer
    frame["final_net_proceeds"] = final.net_proceeds
    accept_q = np.column_stack([
        expit((offer_ledger(frame, frame.initial_offer_ratio * (1 + a)).net_proceeds.to_numpy()
               - reservation) / (0.035 * frame.avm_value.to_numpy()) + 0.35)
        for a in STAGE_ACTIONS
    ])
    delivered = action_indices(frame.buy_action)
    accepted = eligible & (rng.random(n) < accept_q[np.arange(n), delivered])
    closure_draw = rng.random(n) < 0.96
    acquired = (initial | accepted) & closure_draw
    frame["accepted_14d"] = accepted.astype(int)
    frame["completed_acquisition_35d"] = (accepted & closure_draw).astype(int)
    frame["acquired"] = acquired
    frame["acceptance_date"] = pd.NaT
    acceptance_days = rng.integers(1, 15, n)
    acceptance = frame.buy_decision_date + pd.to_timedelta(acceptance_days, "D")
    frame.loc[accepted, "acceptance_date"] = acceptance[accepted]
    initial_dates = frame.offer_date + pd.to_timedelta(1 + (acceptance_days - 1) % 7, "D")
    frame.loc[initial, "acceptance_date"] = initial_dates[initial]
    frame["acquisition_close_date"] = pd.NaT
    initial_close = frame.offer_date + pd.to_timedelta(rng.integers(14, 26, n), "D")
    stage_close = frame.buy_decision_date + pd.to_timedelta(rng.integers(21, 36, n), "D")
    close = initial_close.where(initial, stage_close)
    frame.loc[acquired, "acquisition_close_date"] = close[acquired]
    frame["listing_date"] = frame.acquisition_close_date + pd.Timedelta(days=14)
    frame["sell_decision_date"] = frame.listing_date + pd.Timedelta(days=21)
    frame["sell_features_at"] = frame.sell_decision_date - pd.Timedelta(days=1)
    frame["sell_mature_at"] = frame.sell_decision_date + pd.Timedelta(days=30)
    frame["base_list_price"] = 1.02 * frame.avm_value
    frame["minimum_list_price"] = 0.96 * frame.avm_value
    demand = rng.normal(0, 1, n)
    premium = np.log(frame.base_list_price.to_numpy() / true_value)
    signal = expit(0.7 * demand - 3 * premium + rng.normal(0, 0.6, n))
    early_q = expit(-1.4 + 1.2 * (signal - 0.5) + 0.3 * demand - 8 * premium)
    early = acquired & (rng.random(n) < early_q)
    weak = acquired & ~early & (signal < 0.65)
    frame["early_sold"] = early
    frame["buyer_signal"] = np.where(acquired, signal, np.nan)
    frame["sell_eligible"] = weak
    markdown, sell_probability = randomize(feasible_actions(frame, "sell"), rng)
    frame["sell_action"] = np.where(weak, markdown, 0)
    frame["sell_assignment_probability"] = np.where(weak, sell_probability, np.nan)
    sell_q = np.column_stack([
        expit(-0.4 + 1.8 * (signal - 0.5) + 0.35 * frame.condition.to_numpy()
              + 0.5 * demand - (14 + 8 * frame.condition.to_numpy())
              * np.log(frame.base_list_price.to_numpy() * (1 - a) / true_value))
        for a in STAGE_ACTIONS
    ])
    sell_index = action_indices(frame.sell_action)
    sold = acquired & ~early & (rng.random(n) < sell_q[np.arange(n), sell_index])
    frame["sold_30d"] = np.where(acquired & ~early, sold.astype(float), np.nan)
    post_days = np.where(sold, rng.integers(10, 27, n), 45)
    listing_days = np.where(early, rng.integers(7, 20, n), 21 + post_days)
    frame["resale_close_date"] = frame.listing_date + pd.to_timedelta(listing_days, "D")
    prices = frame.base_list_price.to_numpy() * (1 - frame.sell_action.to_numpy())
    sale_revenue = np.where(early | sold, 0.99 * prices, 0.94 * true_value)
    sale_revenue *= np.exp(rng.normal(-0.02**2 / 2, 0.02, n))
    frame["realized_revenue"] = np.where(acquired, sale_revenue, np.nan)
    frame["repair_cost"] = np.where(
        acquired, frame.repair_forecast * np.exp(rng.normal(-0.18**2 / 2, 0.18, n)), np.nan)
    frame["daily_carrying_cost"] = (
        frame.cash_holding_per_day + 0.08 * frame.company_outlay / 365)
    days = (frame.resale_close_date - frame.acquisition_close_date).dt.days
    frame["realized_disposition_net"] = (
        (1 - SELLING_RATE) * frame.realized_revenue - frame.daily_carrying_cost * days)
    frame["realized_remaining_net"] = np.where(
        weak, (1 - SELLING_RATE) * sale_revenue - frame.daily_carrying_cost * post_days, np.nan)
    frame["lifecycle_contribution"] = np.where(
        acquired, frame.realized_disposition_net - frame.company_outlay - frame.repair_cost, 0)
    frame["lifecycle_contribution"] -= QUOTE_COST * eligible
    truth = pd.DataFrame({
        "home_id": frame.home_id, "true_value": true_value, "private_quality": quality,
        "reservation_value": reservation, "hidden_demand": demand,
    })
    for index, a in enumerate(STAGE_ACTIONS):
        truth[f"buy_close_q_{index}"] = 0.96 * accept_q[:, index]
        truth[f"sell_close_q_{index}"] = sell_q[:, index]
    return frame, truth

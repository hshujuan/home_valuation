"""Explicit feature contracts and strictly prior comparable transactions."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

VALUE_NUMERIC = [
    "log_sqft", "beds", "age", "condition", "market_day",
    "season", "comp_price_per_sqft",
]
CAUSAL_NUMERIC = [
    "log_sqft", "age", "condition", "days_on_market", "views_before",
    "log_base_to_avm", "demand_signal",
]


def preprocessing(numeric):
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler()), numeric),
        ("market", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["market"]),
    ])


def add_comps(query, history, window=40):
    """Exclude same-day and future sales, including a training row's own target."""
    result = query.copy()
    result["log_sqft"] = np.log(result["sqft"] / 1800)
    result["season"] = np.sin(result["market_day"] / 90)
    result["comp_price_per_sqft"] = np.nan
    for market, indices in result.groupby("market").groups.items():
        past = history.loc[history["market"].eq(market)].sort_values("market_day")
        if past.empty:
            continue
        days = past["market_day"].to_numpy()
        rolling = (past["sale_price"] / past["sqft"]).rolling(window, min_periods=1).median()
        positions = np.searchsorted(days, result.loc[indices, "market_day"], side="left") - 1
        valid = positions >= 0
        target_indices = np.asarray(indices)[valid]
        result.loc[target_indices, "comp_price_per_sqft"] = rolling.to_numpy()[positions[valid]]
    return result


def resale_features(frame):
    result = frame.copy()
    if (result[["base_list_price", "avm_value", "sqft"]] <= 0).any().any():
        raise ValueError("Prices, AVM values, and square footage must be positive.")
    result["log_sqft"] = np.log(result["sqft"] / 1800)
    result["log_base_to_avm"] = np.log(result["base_list_price"] / result["avm_value"])
    return result


def action_features(frame, action=None):
    result = resale_features(frame)
    if action is not None:
        result["action"] = action
    if (result["action"] <= -1).any():
        raise ValueError("A price action must leave a strictly positive price.")
    result["log_action"] = np.log1p(result["action"])
    result["action_condition"] = result["log_action"] * result["condition"]
    return result


def mature_rows(frame, outcome="sold_30d"):
    result = frame.loc[frame["label_matured"] & frame[outcome].notna()].copy()
    if result.empty:
        raise ValueError("No mature outcomes are available for estimation.")
    return result

"""Homeowner acceptance and selected-inventory economics."""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss

from .causal import probability_model

SELLER_FEATURES = ["net_to_avm", "belief_to_avm", "attachment_score", "convenience_score",
                   "condition"]


def offer_ledger(frame, offer_ratio):
    result = frame.copy()
    result["offer_ratio"] = offer_ratio
    result["offer"] = result.avm_value * offer_ratio
    result["company_outlay"] = result.offer * (1 - result.service_rate) - result.repair_credit
    result["net_proceeds"] = result.company_outlay - result.seller_closing_cost
    result["net_to_avm"] = result.net_proceeds / result.avm_value
    result["belief_to_avm"] = result.belief_before_offer / result.avm_value
    return result


def fit_seller(frame):
    frame = offer_ledger(frame, frame.offer_ratio).sort_values("market_day")
    train, test = frame.iloc[:int(len(frame) * 0.75)], frame.iloc[int(len(frame) * 0.75):]
    if train.accepted_14d.nunique() < 2:
        raise ValueError("Acceptance-model training requires accepted and declined offers.")
    model = probability_model(SELLER_FEATURES).fit(train, train.accepted_14d)
    prediction = model.predict_proba(test)[:, 1]
    return model, {"brier": float(brier_score_loss(test.accepted_14d, prediction)),
                   "log_loss": float(log_loss(test.accepted_14d, prediction)),
                   "test_n": len(test)}


def acquisition_curves(frame, truth, model, ratios=(0.90, 0.94, 0.98, 1.02)):
    rows = []
    value = truth.true_value.to_numpy()
    for ratio in ratios:
        offer = offer_ledger(frame, ratio)
        q = expit((offer.net_proceeds.to_numpy() - truth.reservation_value.to_numpy()) /
                  (0.025 * offer.avm_value.to_numpy()))
        closing = 0.96 * q
        outlay = offer.company_outlay.to_numpy()
        margin = value * 0.98 - outlay - 9000 - 3500 - 6000
        closing_derivative = 0.96 * q * (1 - q) * (1 - offer.service_rate.to_numpy()) / 0.025
        margin_derivative = -offer.avm_value.to_numpy() * (1 - offer.service_rate.to_numpy())
        selected_value = float(np.average(value, weights=closing))
        rows.append({
            "offer_ratio": ratio, "mean_offer": float(offer.offer.mean()),
            "mean_net_proceeds": float(offer.net_proceeds.mean()),
            "predicted_acceptance": float(model.predict_proba(offer)[:, 1].mean()),
            "oracle_acceptance": float(q.mean()),
            "oracle_closing": float(closing.mean()),
            "selected_true_value": selected_value,
            "selected_avm_overvaluation": float(np.average(
                offer.avm_value.to_numpy() - value, weights=closing)),
            "selection_aware_profit": float(np.mean(closing * margin) - 200),
            "independence_approximation": float(closing.mean() * margin.mean() - 200),
            "joint_slope_per_1pp_offer_ratio": float(
                0.01 * np.mean(closing_derivative * margin + closing * margin_derivative)),
            "factorized_slope_per_1pp_offer_ratio": float(
                0.01 * (closing_derivative.mean() * margin.mean()
                        + closing.mean() * margin_derivative.mean())),
        })
    return pd.DataFrame(rows)


def homeowner_profiles():
    rows = []
    for attachment, convenience, label in [
        (0.01, 0.08, "values convenience"), (0.06, 0.02, "attached/patient"),
        (0.03, 0.04, "middle profile"),
    ]:
        for ratio in [0.90, 0.94, 0.98, 1.02]:
            offer = 500_000 * ratio
            net = offer * 0.98 - 6000 - 3000
            reservation = 500_000 * (0.98 + attachment - convenience)
            rows.append({"profile": label, "asset_value": 500_000, "offer": offer,
                         "net_proceeds": net, "reservation_value": reservation,
                         "acceptance_probability": float(expit((net - reservation) / 12500))})
    return pd.DataFrame(rows)

"""Finite supported actions; acquisition cost is charged once in every state."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, kron

from .simulation import ACTIONS, oracle_resale_q


@dataclass(frozen=True)
class Economics:
    terminal_ratio: float = 0.94
    selling_rate: float = 0.02
    financing_rate: float = 0.08
    mean_early_close_day: float = 18
    terminal_close_day: float = 45
    holding_multiplier: float = 1

    def __post_init__(self):
        if not 0 < self.terminal_ratio <= 1.3:
            raise ValueError("Terminal ratio must be in (0, 1.3].")
        if min(self.selling_rate, self.financing_rate, self.holding_multiplier) < 0:
            raise ValueError("Cost rates and holding multiplier cannot be negative.")
        if not 0 < self.mean_early_close_day <= 30 < self.terminal_close_day:
            raise ValueError("Early and continuation closing times must match the study horizon.")


STRESS_ECONOMICS = (
    ("reference", Economics()),
    ("lower_terminal_value", Economics(terminal_ratio=0.86)),
    ("expensive_holding", Economics(holding_multiplier=5)),
    ("higher_terminal_value", Economics(terminal_ratio=1.01)),
    ("lower_terminal_and_expensive_holding",
     Economics(terminal_ratio=0.86, holding_multiplier=5)),
)


def profit_matrices(frame, q, economics=Economics(), terminal_values=None):
    q = np.asarray(q)
    if q.shape != (len(frame), len(ACTIONS)) or not np.isfinite(q).all():
        raise ValueError("Expected one finite probability per home and supported action.")
    if np.any((q < 0) | (q > 1)):
        raise ValueError("Invalid probabilities; clipping would hide an invalid response model.")
    prices = frame.base_list_price.to_numpy()[:, None] * (1 + ACTIONS)
    early_revenue = prices * 0.99
    terminal = (frame.avm_value.to_numpy() if terminal_values is None else
                np.asarray(terminal_values))[:, None] * economics.terminal_ratio
    revenue = q * early_revenue + (1 - q) * terminal
    basis = frame.cost_basis.to_numpy()[:, None]
    gross = revenue - basis
    days = q * economics.mean_early_close_day + (1 - q) * economics.terminal_close_day
    holding = days * frame.holding_per_day.to_numpy()[:, None] * economics.holding_multiplier
    financing = days * basis * economics.financing_rate / 365
    contribution = gross - economics.selling_rate * revenue - holding - financing
    return {"prices": prices, "q": q, "revenue": revenue, "gross": gross,
            "contribution": contribution, "days": days, "holding": holding,
            "financing": financing}


def choose_actions(values, prices, min_price=None):
    values, prices = np.asarray(values), np.asarray(prices)
    if (values.shape != prices.shape or values.ndim != 2 or not np.isfinite(values).all()
            or not np.isfinite(prices).all() or np.any(prices <= 0)):
        raise ValueError("Finite objective values and prices must have matching two-dimensional shapes.")
    feasible = np.ones(values.shape, dtype=bool)
    if min_price is not None:
        if not np.isfinite(min_price).all():
            raise ValueError("Price floors must be finite.")
        feasible &= prices >= np.asarray(min_price).reshape(-1, 1)
    if np.any(~feasible.any(axis=1)):
        raise ValueError("No feasible supported price for at least one home; abstain.")
    return np.argmax(np.where(feasible, values, -np.inf), axis=1)


def policy_comparison(frame, truth, response, economics=Economics()):
    estimated = profit_matrices(frame, response.matrix(frame), economics)
    worst_case = np.minimum.reduce([
        profit_matrices(frame, estimated["q"], scenario)["contribution"]
        for _, scenario in STRESS_ECONOMICS
    ])
    oracle_q = np.column_stack([oracle_resale_q(frame, truth, a) for a in ACTIONS])
    oracle = profit_matrices(frame, oracle_q, economics, truth.true_value)
    policies = {
        "hold": np.full(len(frame), 2),
        "uniform_3pct_cut": np.full(len(frame), 1),
        "estimated_gross_optimum": choose_actions(estimated["gross"], estimated["prices"]),
        "estimated_contribution_optimum": choose_actions(
            estimated["contribution"], estimated["prices"]),
        "stress_robust_contribution": choose_actions(worst_case, estimated["prices"]),
        "oracle_contribution_optimum": choose_actions(oracle["contribution"], oracle["prices"]),
    }
    indices = np.arange(len(frame))
    oracle_best = oracle["contribution"].max(axis=1)
    rows = []
    for name, choices in policies.items():
        reward = oracle["contribution"][indices, choices]
        rows.append({"policy": name, "oracle_sale_probability": float(oracle_q[indices, choices].mean()),
                     "oracle_gross_profit": float(oracle["gross"][indices, choices].mean()),
                     "oracle_contribution": float(reward.mean()),
                     "oracle_regret": float(np.mean(oracle_best - reward)),
                     "mean_action": float(ACTIONS[choices].mean())})
    return pd.DataFrame(rows), policies, estimated


def example_curve(frame, response, economics=Economics()):
    if len(frame) != 1:
        raise ValueError("Select exactly one home for the individual profile illustration.")
    values = profit_matrices(frame, response.matrix(frame), economics)
    q = values["q"][0]
    return pd.DataFrame({
        "action": ACTIONS, "list_price": values["prices"][0], "sale_probability": q,
        "change_pp_vs_hold": (q - q[2]) * 100,
        "relative_change_vs_hold": (q / q[2] - 1) if q[2] > 0 else np.nan,
        "expected_gross_profit": values["gross"][0],
        "expected_gross_margin_rate": values["gross"][0] / values["revenue"][0],
        "expected_contribution": values["contribution"][0],
        "expected_holding_days": values["days"][0],
    })


def policy_weights(n, choices):
    """Represent a deterministic policy or a lottery over supported prices."""
    choices = np.asarray(choices)
    if choices.shape == (n,):
        if not np.isin(choices, np.arange(len(ACTIONS))).all():
            raise ValueError("The policy must select one supported action index per row.")
        return np.eye(len(ACTIONS))[choices.astype(int)]
    if (choices.shape != (n, len(ACTIONS)) or not np.isfinite(choices).all()
            or np.any(choices < 0)
            or not np.allclose(choices.sum(axis=1), 1, rtol=0, atol=1e-8)):
        raise ValueError("Policy probabilities must be nonnegative and sum to one per home.")
    return choices


def off_policy_scores(frame, choices, estimated):
    """Holdout DR scores, including genuine lotteries, not averaged list prices."""
    n = len(frame)
    if n < 2:
        raise ValueError("At least two holdout homes are required for policy inference.")
    weights = policy_weights(n, choices)
    if not frame.design.eq("multiarm").all():
        raise ValueError("This evaluator requires randomized multi-arm logged data.")
    observed, closed = pd.to_datetime(frame.observed_through), pd.to_datetime(frame.close_date)
    if observed.isna().any() or closed.isna().any() or (observed < closed).any():
        raise ValueError("Lifecycle policy evaluation requires mature closing rewards.")
    propensity = frame.assignment_probability.to_numpy()
    if not np.isfinite(propensity).all() or np.any((propensity <= 0) | (propensity > 1)):
        raise ValueError("Positive known logging probabilities are required.")
    matches = np.isclose(frame.action.to_numpy()[:, None], ACTIONS[None, :])
    if not np.all(matches.sum(axis=1) == 1):
        raise ValueError("Logged actions must belong to the supported action set.")
    predicted = np.asarray(estimated["contribution"])
    if predicted.shape != weights.shape or not np.isfinite(predicted).all():
        raise ValueError("One finite reward prediction per home/action is required.")
    assigned = matches.argmax(axis=1)
    importance = weights[np.arange(n), assigned] / propensity
    reward = frame.realized_contribution.to_numpy()
    if not np.isfinite(reward).all():
        raise ValueError("Lifecycle rewards must be finite and observed.")
    scores = (weights * predicted).sum(axis=1) + importance * (
        reward - predicted[np.arange(n), assigned])
    return scores


def off_policy_value(frame, choices, estimated):
    scores = off_policy_scores(frame, choices, estimated)
    n = len(frame)
    se = scores.std(ddof=1) / np.sqrt(n)
    weights = policy_weights(n, choices)
    matches = np.isclose(frame.action.to_numpy()[:, None], ACTIONS[None, :])
    return {"dr_value": float(scores.mean()), "lower": float(scores.mean() - 1.96 * se),
            "upper": float(scores.mean() + 1.96 * se),
            "matching_actions": int(((weights * matches).sum(axis=1) > 0).sum()),
            "outcome": "Lifecycle contribution under the specified day-45 terminal sale policy"}


def paired_policy_difference(frame, choices, baseline, estimated):
    delta = (off_policy_scores(frame, choices, estimated)
             - off_policy_scores(frame, baseline, estimated))
    se = delta.std(ddof=1) / np.sqrt(len(delta))
    return {"dr_difference_vs_hold": float(delta.mean()), "difference_se": float(se),
            "difference_lower": float(delta.mean() - 1.96 * se),
            "difference_upper": float(delta.mean() + 1.96 * se)}


def optimize_completion_target(values, q, prices, target, min_price=None):
    """Maximize contribution with an expected portfolio completion constraint."""
    values, q, prices = np.asarray(values), np.asarray(q), np.asarray(prices)
    choose_actions(values, prices, min_price)
    n, actions = values.shape
    if (n == 0 or actions != len(ACTIONS) or q.shape != values.shape
            or not np.isfinite(q).all() or np.any((q < 0) | (q > 1))):
        raise ValueError("Provide valid completion probabilities for every supported action.")
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("The completion target must be a probability in [0, 1].")
    feasible = np.ones(values.shape, dtype=bool)
    if min_price is not None:
        feasible &= prices >= np.asarray(min_price).reshape(-1, 1)
    maximum = float(np.where(feasible, q, -np.inf).max(axis=1).mean())
    if target > maximum + 1e-12:
        raise ValueError(f"Infeasible completion target; supported maximum is {maximum:.6f}.")
    if abs(target - maximum) <= 1e-12:
        best_q = np.where(feasible, q, -np.inf).max(axis=1, keepdims=True)
        choices = np.where(feasible & (q == best_q), values, -np.inf).argmax(axis=1)
        weights = policy_weights(n, choices)
        shadow_price = None
    else:
        result = linprog(
            -values.ravel(), A_ub=csr_matrix(-q.reshape(1, -1)), b_ub=[-n * target],
            A_eq=kron(eye(n, format="csr"), np.ones((1, actions)), format="csr"),
            b_eq=np.ones(n), bounds=np.column_stack([np.zeros(n * actions), feasible.ravel()]),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Portfolio optimization failed: {result.message}")
        weights = policy_weights(n, result.x.reshape(n, actions))
        shadow_price = float(-result.ineqlin.marginals[0])
    completion = float((weights * q).sum() / n)
    if completion < target - 1e-8 or np.any(weights[~feasible] > 1e-8):
        raise RuntimeError("Solver output violates the supported-price/volume contract.")
    return {
        "weights": weights, "target_completion": float(target),
        "predicted_completion": completion,
        "predicted_contribution": float((weights * values).sum() / n),
        "max_feasible_completion": maximum,
        "shadow_price_per_expected_completion": shadow_price,
        "randomized_home_fraction": float(np.mean(weights.max(axis=1) < 1 - 1e-8)),
    }


def completion_frontier(frame, truth, estimated):
    q = estimated["q"]
    hold_rate, maximum = float(q[:, 2].mean()), float(q.max(axis=1).mean())
    oracle_q = np.column_stack([oracle_resale_q(frame, truth, a) for a in ACTIONS])
    oracle = profit_matrices(frame, oracle_q, terminal_values=truth.true_value)
    baseline = np.full(len(frame), 2)
    rows = []
    for label, target in [
        ("unconstrained", 0), ("hold_rate_target", hold_rate),
        ("half_remaining_headroom", (hold_rate + maximum) / 2),
        ("maximum_supported_completion", maximum),
    ]:
        solution = optimize_completion_target(
            estimated["contribution"], q, estimated["prices"], target)
        weights = solution.pop("weights")
        rows.append({
            "policy": label, **solution,
            "oracle_completion": float((weights * oracle_q).sum(axis=1).mean()),
            "oracle_contribution": float((weights * oracle["contribution"]).sum(axis=1).mean()),
            **off_policy_value(frame, weights, estimated),
            **paired_policy_difference(frame, weights, baseline, estimated),
        })
    return pd.DataFrame(rows)

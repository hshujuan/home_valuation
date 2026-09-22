import numpy as np
import pandas as pd
import pytest

from home_valuation.instruments import estimate_iv, exclusion_sensitivity, instrument_balance
from home_valuation.optimization import (
    off_policy_scores, optimize_completion_target, paired_policy_difference,
    policy_weights, profit_matrices,
)
from home_valuation.simulation import iv_sample


def test_iv_sensitivity_matches_moment_identity():
    frame, _ = iv_sample(1200, 91)
    base = estimate_iv(frame)
    sensitivity = exclusion_sensitivity(frame, [-0.01, 0, 0.01])
    assert np.allclose(sensitivity.adjusted_effect,
                       base["estimate"] - sensitivity.assumed_direct_effect / base["first_stage"])
    at_zero = sensitivity.iloc[1]
    assert at_zero.lower == pytest.approx(base["lower"])
    assert at_zero.ar_confidence_set == base["ar_confidence_set"]
    with pytest.raises(ValueError, match="finite"):
        exclusion_sensitivity(frame, [np.nan])


def test_assignment_check_does_not_use_treatment_or_outcome():
    frame, _ = iv_sample(600, 91)
    expected = instrument_balance(frame)
    assert instrument_balance(frame.assign(cut=1, sold_30d=0)) == expected


def test_volume_optimizer_known_fractional_solution_and_shadow_price():
    values = np.array([[100, 80, 0, 0], [60, 40, 0, 0]])
    q = np.array([[0.2, 0.6, 0, 0], [0.3, 0.9, 0, 0]])
    prices = np.array([[940, 970, 1000, 1030]] * 2)
    optimum = optimize_completion_target(values, q, prices, 0.5)
    assert optimum["predicted_completion"] == pytest.approx(0.5)
    assert optimum["predicted_contribution"] == pytest.approx(215 / 3)
    assert optimum["weights"][1, 1] == pytest.approx(5 / 6)
    assert optimum["shadow_price_per_expected_completion"] == pytest.approx(100 / 3)
    tighter = optimize_completion_target(values, q, prices, 0.5001)
    loss = (optimum["predicted_contribution"] - tighter["predicted_contribution"]) / 0.0001
    assert loss == pytest.approx(optimum["shadow_price_per_expected_completion"])
    endpoint = optimize_completion_target(values, q, prices, 0.75)
    assert endpoint["predicted_completion"] == pytest.approx(0.75)
    assert endpoint["shadow_price_per_expected_completion"] is None
    assert np.all((endpoint["weights"] == 0) | (endpoint["weights"] == 1))
    with pytest.raises(ValueError, match="Infeasible"):
        optimize_completion_target(values, q, prices, 0.8)
    with pytest.raises(ValueError, match="No feasible"):
        optimize_completion_target(values, q, prices, 0.2, min_price=[2000, 2000])


def test_stochastic_ope_matches_hand_computation(study):
    frame = study.tables["resale_multiarm"].head(20)
    expected = profit_matrices(frame, study.models["response"].matrix(frame))
    weights = np.broadcast_to([0.25] * 4, (len(frame), 4))
    action_index = np.array([np.flatnonzero(np.isclose(a, [-0.06, -0.03, 0, 0.03]))[0]
                             for a in frame.action])
    hand = (expected["contribution"].mean(axis=1) + frame.realized_contribution.to_numpy()
            - expected["contribution"][np.arange(len(frame)), action_index])
    assert np.allclose(off_policy_scores(frame, weights, expected), hand)
    with pytest.raises(ValueError, match="sum to one"):
        policy_weights(len(frame), np.ones((len(frame), 4)))


def test_paired_policy_identity_and_immature_reward_rejection(study):
    frame = study.tables["resale_multiarm"].head(40).copy()
    expected = profit_matrices(frame, study.models["response"].matrix(frame))
    hold = np.full(len(frame), 2)
    same = paired_policy_difference(frame, hold, hold, expected)
    assert all(value == 0 for value in same.values())
    cut = np.ones(len(frame), dtype=int)
    forward = paired_policy_difference(frame, cut, hold, expected)
    reverse = paired_policy_difference(frame, hold, cut, expected)
    assert forward["dr_difference_vs_hold"] == pytest.approx(-reverse["dr_difference_vs_hold"])
    assert forward["difference_se"] == pytest.approx(reverse["difference_se"])
    missing = frame.copy()
    missing.loc[missing.index[0], "close_date"] = pd.NaT
    with pytest.raises(ValueError, match="mature"):
        off_policy_scores(missing, hold, expected)
    frame["observed_through"] = pd.to_datetime(frame.decision_date)
    with pytest.raises(ValueError, match="mature"):
        off_policy_scores(frame, hold, expected)


def test_frontier_feasibility_and_order(study):
    frontier = study.tables["portfolio_frontier"]
    assert np.all(frontier.predicted_completion >= frontier.target_completion - 1e-8)
    assert np.all(np.diff(frontier.predicted_contribution) <= 1e-6)
    assert np.all(frontier.randomized_home_fraction <= 1 / study.metadata["response_test_n"] + 1e-8)

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from home_valuation.causal import difference_in_means, doubly_robust
from home_valuation.features import add_comps, mature_rows
from home_valuation.instruments import ar_confidence_set, quadratic_set
from home_valuation.optimization import Economics, choose_actions, profit_matrices
from home_valuation.simulation import ACTIONS, iv_sample, market_sales, oracle_resale_q
from home_valuation.seller import acquisition_curves


def test_seed_reproducibility():
    a, ta = market_sales(400, 21)
    b, tb = market_sales(400, 21)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ta, tb)


def test_comps_exclude_same_day_and_future():
    history = pd.DataFrame({"market": ["M"]*3, "market_day": [1, 2, 3],
                            "sqft": [1000]*3, "sale_price": [100_000, 900_000, 9_000_000]})
    query = history.iloc[[1]].copy()
    assert add_comps(query, history).comp_price_per_sqft.iloc[0] == 100
    history.loc[history.market_day >= 2, "sale_price"] *= 100
    assert add_comps(query, history).comp_price_per_sqft.iloc[0] == 100


def test_oracle_and_time_contracts(study):
    frame, truth = study.tables["resale_randomized"], study.truth["resale"]
    q = np.column_stack([oracle_resale_q(frame, truth, a) for a in ACTIONS])
    assert (np.diff(q, axis=1) <= 0).all()
    assert np.allclose(frame.loc[frame.cut.eq(1), "list_price"],
                       0.97 * frame.loc[frame.cut.eq(1), "base_list_price"])
    assert (frame.listing_date < frame.decision_date).all()
    assert (frame.condition_verified_at < frame.decision_date).all()
    assert (frame.contract_date <= frame.close_date).all()
    elapsed = (frame.close_date - frame.decision_date).dt.days
    assert ((elapsed <= 30) == frame.sold_30d.astype(bool)).all()
    assert (frame.loc[frame.sold_30d.eq(0), "terminal_sale_price"] > 0).all()


def test_future_and_oracle_columns_do_not_change_predictions(study):
    frame = study.tables["resale_multiarm"].head(5).copy()
    before = study.models["response"].matrix(frame)
    frame["hidden_demand"] = 1e12
    frame["sold_30d"] = 1 - frame.sold_30d
    frame["sale_price_30d"] = 1
    assert np.array_equal(before, study.models["response"].matrix(frame))


def test_unmatured_labels_not_negative(study):
    frame = study.tables["resale_randomized"].head(20).copy()
    frame.loc[frame.index[:5], "label_matured"] = False
    assert len(mature_rows(frame)) == 15
    frame["label_matured"] = False
    with pytest.raises(ValueError, match="No mature"):
        mature_rows(frame)


def test_empty_treatment_arm_explicit(study):
    frame = study.tables["resale_randomized"].copy()
    frame["cut"] = 0
    with pytest.raises(ValueError):
        difference_in_means(frame)
    with pytest.raises(ValueError):
        doubly_robust(frame)


def test_unsupported_price_and_no_feasible_action(study):
    with pytest.raises(ValueError, match="Unsupported"):
        study.models["response"].predict(study.models["profile"], -0.20)
    with pytest.raises(ValueError, match="No feasible"):
        choose_actions(np.array([[1, 2]]), np.array([[400, 500]]), [600])


def test_cost_basis_once_and_sunk(study):
    frame = study.models["profile"].copy()
    q = study.models["response"].matrix(frame)
    before = profit_matrices(frame, q)
    frame["cost_basis"] += 100_000
    after = profit_matrices(frame, q)
    assert np.allclose(before["gross"] - after["gross"], 100_000)
    assert np.array_equal(before["gross"].argmax(axis=1), after["gross"].argmax(axis=1))
    assert (before["revenue"] > 0).all()
    with pytest.raises(ValueError, match="Invalid probabilities"):
        profit_matrices(frame, np.full((1, 4), 1.1))
    with pytest.raises(ValueError):
        Economics(terminal_close_day=15)


def test_ar_inversion_matches_robust_test():
    frame, _ = iv_sample(n=1200, seed=70)
    intervals = ar_confidence_set(frame)
    x = sm.add_constant(frame[["x", "recommendation"]])
    for beta in np.linspace(-1, 1, 35):
        fit = sm.OLS(frame.sold_30d - beta * frame.cut, x).fit(cov_type="HC1")
        accepted = fit.pvalues["recommendation"] >= 0.05
        in_set = any(lo <= beta <= hi for lo, hi in intervals)
        assert accepted == in_set


def test_unbounded_sets_not_clipped():
    intervals = quadratic_set(-1, 0, 1)
    assert intervals == [(-np.inf, -1.0), (1.0, np.inf)]
    assert quadratic_set(0, 0, -1) == [(-np.inf, np.inf)]
    assert quadratic_set(1, 0, 1) == []


def test_selection_and_policy_evaluation(study):
    acquisition = study.tables["acquisition_curves"]
    assert (acquisition.selection_aware_profit -
            acquisition.independence_approximation).abs().max() > 100
    policies = study.tables["policy_comparison"]
    assert (policies.oracle_regret >= -1e-8).all()
    oracle = policies.loc[policies.policy.eq("oracle_contribution_optimum")]
    assert oracle.oracle_regret.iloc[0] == pytest.approx(0)
    assert study.metadata["valuation"]["train_last_day"] < study.metadata["valuation"]["test_first_day"]


def test_three_percent_is_not_three_ratio_points():
    ratio = 0.94
    assert ratio * 0.97 == pytest.approx(0.9118)
    assert not np.isclose(ratio * 0.97, ratio - 0.03)


def test_selection_aware_derivative_matches_finite_difference(study):
    epsilon = 1e-5
    curves = acquisition_curves(
        study.tables["seller_offers"], study.truth["seller"], study.models["seller"],
        ratios=(0.94 - epsilon, 0.94, 0.94 + epsilon))
    numerical = (curves.selection_aware_profit.iloc[2] -
                 curves.selection_aware_profit.iloc[0]) / (2 * epsilon) * 0.01
    assert numerical == pytest.approx(curves.joint_slope_per_1pp_offer_ratio.iloc[1], rel=1e-6)


def test_stress_policy_uses_supported_maximin_action(study):
    from home_valuation.optimization import STRESS_ECONOMICS, policy_comparison
    frame = study.tables["resale_multiarm"].head(10)
    truth = study.truth["resale"].loc[frame.index]
    response = study.models["response"]
    _, policies, estimated = policy_comparison(frame, truth, response)
    worst = np.minimum.reduce([profit_matrices(frame, estimated["q"], economics)["contribution"]
                               for _, economics in STRESS_ECONOMICS])
    assert np.array_equal(policies["stress_robust_contribution"], worst.argmax(axis=1))

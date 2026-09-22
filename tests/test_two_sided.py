from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from home_valuation.two_sided import (
    continuation_rewards, fit_continuation, fit_stage_response, require_maturity,
    run_two_sided, save_two_sided, select_escalation, stage_features, stage_scores,
)
from home_valuation.two_sided_simulation import (
    QUOTE_COST, STAGE_ACTIONS, TwoSidedConfig, feasible_actions, generate_cohort,
)


@pytest.fixture(scope="module")
def linked():
    return run_two_sided(TwoSidedConfig(n_market=1200, n_development=3600,
                                       n_continuation=3600, n_evaluation=3600))


def test_linked_ids_timing_and_reward_accounting(linked):
    seen = set()
    previous_maturity = pd.Timestamp.min
    for name in ["development", "continuation", "evaluation"]:
        frame = linked.tables[name]
        assert not seen.intersection(frame.home_id)
        seen.update(frame.home_id)
        assert previous_maturity < frame.offer_date.min()
        previous_maturity = frame.reward_mature_at.max()
        acquired = frame.loc[frame.acquired]
        weak = frame.loc[frame.sell_eligible]
        accept = frame.loc[frame.accepted_14d.eq(1)]
        initial = frame.loc[frame.initial_accepted_7d.eq(1)]
        assert ((initial.acceptance_date - initial.offer_date).dt.days.between(1, 7)).all()
        assert ((accept.acceptance_date - accept.buy_decision_date).dt.days.between(1, 14)).all()
        closes = frame.loc[frame.completed_acquisition_35d.eq(1)]
        assert ((closes.acquisition_close_date - closes.buy_decision_date).dt.days <= 35).all()
        assert (closes.acquisition_close_date >= closes.acceptance_date).all()
        assert (frame.completed_acquisition_35d <= frame.accepted_14d).all()
        assert (weak.acquired & ~weak.early_sold & (weak.buyer_signal < 0.65)).all()
        assert (acquired.acquisition_close_date < acquired.listing_date).all()
        assert (acquired.resale_close_date <= acquired.observed_through).all()
        elapsed = (weak.resale_close_date - weak.sell_decision_date).dt.days
        assert np.array_equal(elapsed <= 30, weak.sold_30d.astype(bool))
        assert (elapsed[weak.sold_30d.eq(0)] == 45).all()
        expected = (acquired.realized_disposition_net - acquired.company_outlay
                    - acquired.repair_cost - QUOTE_COST * acquired.buy_eligible)
        assert np.allclose(expected, acquired.lifecycle_contribution)
        assert (frame.loc[~frame.acquired & frame.buy_eligible, "lifecycle_contribution"]
                == -QUOTE_COST).all()
        assert not any("true_" in c or "hidden_" in c for c in frame.columns)


def test_stage_features_and_policies_ignore_future_and_oracle_columns(linked):
    frame = linked.tables["evaluation"].loc[lambda d: d.buy_eligible].head(40).copy()
    model = linked.models["continuation"]["optimized"]
    before = model.matrix(frame)
    allowed = stage_features(frame, "buy", 0).columns
    for column in ["company_outlay", "realized_revenue", "realized_disposition_net",
                   "repair_cost", "buyer_signal", "accepted_14d", "sold_30d",
                   "lifecycle_contribution", "true_value", "hidden_demand"]:
        assert column not in allowed
        frame[column] = 1e12
    assert np.array_equal(before, model.matrix(frame))
    asof_only = frame.drop(columns=[
        "company_outlay", "realized_revenue", "realized_disposition_net", "repair_cost",
        "buyer_signal", "accepted_14d", "sold_30d", "lifecycle_contribution",
        "base_list_price", "sell_action", "sell_eligible", "acquired",
    ])
    assert np.array_equal(before, model.matrix(asof_only))
    frame["buy_features_at"] = frame.buy_decision_date + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="before"):
        model.matrix(frame)


def test_chosen_gain_hurdle_and_feasibility():
    values = np.array([[100, 260, 330, 400], [100, 200, 320, 330], [100, 110, 130, 140]])
    feasible = np.ones_like(values, dtype=bool)
    feasible[1, 3] = False
    choice, gain = select_escalation(values, feasible, min_gain=150, tolerance=150)
    assert np.array_equal(choice, [1, 2, 0])
    assert np.array_equal(gain, [160, 220, 0])
    assert np.all(gain[choice > 0] >= 150)
    with pytest.raises(ValueError, match="nonnegative"):
        select_escalation(values, feasible, tolerance=-1)


def test_randomization_and_conditional_support(linked):
    frame = linked.tables["evaluation"].loc[lambda d: d.buy_eligible].copy()
    allowed = feasible_actions(frame, "buy")
    assert np.allclose(frame.buy_assignment_probability, 1 / allowed.sum(axis=1))
    restricted = frame.loc[frame.max_escalation.eq(0)].head(5)
    assert len(restricted)
    with pytest.raises(ValueError, match="support"):
        stage_scores(restricted, np.ones(len(restricted), int),
                     np.zeros((len(restricted), 4)), np.zeros(len(restricted)), "buy")
    with pytest.raises(ValueError, match="Unsupported"):
        stage_features(frame, "buy", 0.04)


def test_sequential_scores_recover_a_known_two_action_path():
    a, b = np.repeat(np.arange(4), 4), np.tile(np.arange(4), 4)
    frame = pd.DataFrame({
        "max_escalation": 0.03, "base_list_price": 100.0, "minimum_list_price": 90.0,
        "buy_action": STAGE_ACTIONS[a], "sell_action": STAGE_ACTIONS[b],
        "buy_assignment_probability": 0.25, "sell_assignment_probability": 0.25,
    })
    reward = 100 + 10 * a + b
    downstream, _ = stage_scores(
        frame, np.full(16, 2), np.full((16, 4), 3.0), reward, "sell")
    scores, _ = stage_scores(
        frame, np.full(16, 1), np.full((16, 4), 9.0), downstream, "buy")
    assert scores.mean() == pytest.approx(112)


def test_maturity_and_missing_rewards_are_not_silently_zero(linked):
    frame = linked.tables["development"].loc[lambda d: d.buy_eligible].copy()
    frame.loc[frame.index[0], "completed_acquisition_35d"] = np.nan
    with pytest.raises(ValueError, match="observed outcomes"):
        fit_stage_response(frame, "buy")
    frame["observed_through"] = frame.buy_mature_at - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Fully observed"):
        fit_stage_response(frame, "buy")
    frame = linked.tables["continuation"].loc[lambda d: d.buy_eligible].copy()
    acquired_id = frame.loc[frame.acquired].index[0]
    frame.loc[acquired_id, "resale_close_date"] = pd.NaT
    with pytest.raises(ValueError, match="mature disposition"):
        continuation_rewards(frame, linked.models["sell_response"], "hold")
    frame["observed_through"] = pd.NaT
    with pytest.raises(ValueError, match="missing"):
        require_maturity(frame, "lifecycle")


def test_downstream_policy_really_changes_acquisition_learning(linked):
    frame = linked.tables["continuation"].loc[lambda d: d.buy_eligible].copy()

    class SteepResponse:
        def matrix(self, data):
            return np.tile([0.01, 0.05, 0.15, 0.95], (len(data), 1))

    hold, _ = continuation_rewards(frame, SteepResponse(), "hold")
    optimized, _ = continuation_rewards(frame, SteepResponse(), "optimized")
    assert np.max(np.abs(optimized - hold)) > 100
    assert np.all(hold[~frame.acquired] == -QUOTE_COST)
    first, second = fit_continuation(frame, hold), fit_continuation(frame, optimized)
    assert not np.allclose(first.matrix(frame.head(30)), second.matrix(frame.head(30)))


def test_new_reports_do_not_overwrite_baseline_and_gains_reconcile(linked, tmp_path):
    sentinel = tmp_path / "reference_results.md"
    sentinel.write_text("preserve original study", encoding="utf-8")
    save_two_sided(linked, tmp_path)
    assert sentinel.read_text() == "preserve original study"
    assert (tmp_path / "two_sided_results.md").exists()
    decisions = linked.tables["buy_decisions"]
    assert np.allclose(
        decisions.predicted_incremental_value,
        decisions.predicted_joint_value_chosen - decisions.predicted_joint_value_hold)
    assert (decisions.chosen_escalation <= decisions.max_escalation + 1e-12).all()
    baseline = linked.tables["policy_evaluation"].iloc[0]
    assert baseline.difference_vs_hold_both == baseline.difference_lower == baseline.difference_upper == 0
    cohort = linked.tables["cohort_policy_evaluation"].iloc[0]
    assert cohort.n_original_prospects == len(linked.tables["evaluation"])
    assert cohort.difference_vs_hold_both == 0
    candidates = linked.tables["buy_candidate_values"]
    assert candidates.loc[~candidates.feasible, "joint_value_optimized_resale"].isna().all()


def test_whole_cohort_value_reconciles_with_eligible_and_fixed_paths(linked):
    frame = linked.tables["evaluation"]
    for policy in linked.tables["cohort_policy_evaluation"].itertuples():
        rule = "hold" if policy.policy == "hold_both" else "optimized"
        downstream, _ = continuation_rewards(frame, linked.models["sell_response"], rule)
        assert np.all(downstream[~frame.acquired & ~frame.buy_eligible] == 0)
        eligible = linked.tables["policy_evaluation"].set_index("policy").loc[policy.policy]
        expected = (eligible.dr_value * eligible.n_eligible_sellers
                    + downstream[~frame.buy_eligible].sum()) / len(frame)
        assert policy.dr_value_per_original_prospect == pytest.approx(expected)


def test_cohort_generator_is_seeded_without_python_hash(study):
    history, avm = study.tables["market_sales"], study.models["avm"]
    a, ta = generate_cohort(500, 98, "probe", 760, history, avm)
    b, tb = generate_cohort(500, 98, "probe", 760, history, avm)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ta, tb)


@pytest.mark.parametrize("command", ["pipeline", "two-sided"])
def test_cli_keeps_original_defaults_and_separate_extension_output(command, monkeypatch):
    from home_valuation import __main__ as cli
    from home_valuation import two_sided

    called = {}
    result = SimpleNamespace(tables={
        "causal_effects": pd.DataFrame(columns=["design", "method", "estimate", "oracle_ate"]),
        "policy_evaluation": pd.DataFrame(),
    })

    def run(config, *resampling):
        called["config"], called["resampling"] = config, resampling
        return result

    def save(study, output, public_report):
        assert study is result
        called["output"], called["public_report"] = output, public_report

    monkeypatch.setattr("sys.argv", ["home_valuation", command])
    monkeypatch.setattr(cli, "run_study", run)
    monkeypatch.setattr(cli, "save_study", save)
    monkeypatch.setattr(two_sided, "run_two_sided", run)
    monkeypatch.setattr(two_sided, "save_two_sided", save)
    cli.main()
    assert called["config"].seed == 42
    assert called["config"].n_market == 5000
    assert not called["public_report"]
    if command == "pipeline":
        assert called["output"] == Path("outputs") / "reference"
        assert called["resampling"] == (30, 30)
        assert called["config"].n_sellers == 4000
        assert called["config"].n_resale == 6000
        assert called["config"].n_iv == 8000
    else:
        assert called["output"] == Path("outputs") / "two_sided"
        assert called["resampling"] == ()
        assert called["config"].n_development == 12000
        assert called["config"].n_continuation == 10000
        assert called["config"].n_evaluation == 8000


@pytest.mark.parametrize("flags", [["--size", "2399"], ["--bootstrap", "0"], ["--monte-carlo", "0"]])
def test_linked_cli_rejects_unsupported_configuration(flags, monkeypatch):
    from home_valuation.__main__ import main

    monkeypatch.setattr("sys.argv", ["home_valuation", "two-sided", *flags])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2

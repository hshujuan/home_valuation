"""Sanity checks: the causal estimators recover the planted truth, the naive ones do not."""
import pytest

from offerpad_simulator.config import SimConfig
from offerpad_simulator.simulate import simulate
from offerpad_simulator.causal import seller_side_estimates, buyer_side_estimates
from offerpad_simulator.optimize import buyer_policy_curves


@pytest.fixture(scope="module")
def df():
    return simulate(SimConfig(n_leads=30_000, seed=3))


@pytest.fixture(scope="module")
def seller(df):
    return seller_side_estimates(df, n_boot=2).set_index("estimator")


def test_randomized_iv_recovers_truth(seller):
    r = seller.loc["2SLS: randomized spread arm"]
    assert abs(r["estimate"] - r["truth"]) < 3 * r["se"]


def test_naive_is_badly_biased(seller):
    r = seller.loc["Naive OLS (LPM)"]
    assert abs(r["bias"]) > 0.5 * abs(r["truth"])


def test_invalid_instrument_is_flagged(seller):
    note = seller.loc["2SLS: valid + rate (J-test)", "note"]
    assert float(note.split("= ")[1]) < 0.05


def test_buyer_side_theta(df):
    b = buyer_side_estimates(df).set_index("estimator")
    r = b.loc["2SLS: log DOM (theta/100)"]
    assert abs(r["estimate"] - r["truth"]) < 3 * r["se"]


def test_causal_planner_beats_naive(df):
    o = buyer_policy_curves(df)["optimum"].set_index("planner")
    assert o.loc["causal (2SLS)", "regret_vs_truth_opt"] < o.loc["naive (OLS)", "regret_vs_truth_opt"]

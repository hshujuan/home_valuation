import pytest

from home_valuation.pipeline import run_study
from home_valuation.simulation import Config


@pytest.fixture(scope="session")
def study():
    return run_study(Config(seed=42, n_market=1200, n_sellers=1200,
                            n_resale=1200, n_iv=1800))

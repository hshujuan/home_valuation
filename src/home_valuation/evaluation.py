"""Statistical checks and sensitivity analyses, separate from model fitting."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .causal import fit_response
from .instruments import estimate_iv
from .optimization import STRESS_ECONOMICS, example_curve
from .simulation import iv_sample
from .valuation import error_metrics


def probability_metrics(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    return {
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "actual_rate": float(y.mean()), "predicted_rate": float(probability.mean()),
    }


def calibration_table(y, probability):
    actual, predicted = calibration_curve(y, probability, n_bins=8, strategy="quantile")
    return pd.DataFrame({"predicted_probability": predicted, "observed_rate": actual})


def iv_monte_carlo(repetitions=30, n=3000, seed=1000):
    rows = []
    for iteration in range(repetitions):
        frame, truth = iv_sample(n, seed + iteration, "strong")
        result = estimate_iv(frame)
        rows.append({"seed": seed + iteration, "estimate": result["estimate"],
                     "oracle_late": truth["oracle_late"],
                     "error": result["estimate"] - truth["oracle_late"],
                     "covered": result["lower"] <= truth["oracle_late"] <= result["upper"]})
    return pd.DataFrame(rows)


def bootstrap_curve(train, profile, repetitions=30, seed=9000):
    """Pointwise response uncertainty conditional on a frozen AVM; not a joint guarantee."""
    if repetitions < 10:
        raise ValueError("Use at least ten resamples for the illustrative uncertainty plot.")
    rng = np.random.default_rng(seed)
    rows = []
    for repetition in range(repetitions):
        boot = train.iloc[rng.integers(0, len(train), len(train))]
        model = fit_response(boot)
        curve = example_curve(profile, model)
        curve["resample"] = repetition
        rows.append(curve)
    samples = pd.concat(rows, ignore_index=True)
    bounds = samples.groupby("action").agg(
        q_lower=("sale_probability", lambda x: x.quantile(0.025)),
        q_upper=("sale_probability", lambda x: x.quantile(0.975)),
        contribution_lower=("expected_contribution", lambda x: x.quantile(0.025)),
        contribution_upper=("expected_contribution", lambda x: x.quantile(0.975)),
    ).reset_index()
    return bounds


def stress_curves(profile, model):
    rows = []
    for label, economics in STRESS_ECONOMICS:
        curve = example_curve(profile, model, economics)
        curve["scenario"] = label
        rows.append(curve)
    return pd.concat(rows, ignore_index=True)


def valuation_slices(predictions):
    frame = predictions.copy()
    frame["price_quartile"] = pd.qcut(frame.sale_price, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    rows = []
    for dimension in ["market", "price_quartile"]:
        for label, subset in frame.groupby(dimension, observed=True):
            covered = (subset.sale_price >= subset.lower) & (subset.sale_price <= subset.upper)
            rows.append({"dimension": dimension, "slice": str(label), "n": len(subset),
                         **error_metrics(subset.sale_price, subset.prediction),
                         "interval_coverage": float(covered.mean())})
    return pd.DataFrame(rows)


def condition_reporting_ablation(history, avm, prepared_test):
    verified = prepared_test.copy()
    reported = prepared_test.copy()
    reported["condition"] = reported["reported_condition"]
    actual = verified.sale_price.to_numpy()
    return pd.DataFrame([
        {"input": label, "mae": float(np.mean(np.abs(avm.predict(frame) - actual)))}
        for label, frame in [("verified condition", verified), ("optimistic self-report", reported)]
    ])

"""IV diagnostics and exact quadratic inversion of a one-instrument robust AR test."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from scipy.special import expit
from scipy.stats import chi2


def quadratic_set(a, b, c):
    """Intervals satisfying a*x**2 + b*x + c <= 0; retain unbounded sets."""
    tolerance = 1e-12 * max(1, abs(a), abs(b), abs(c))
    if abs(a) < tolerance:
        if abs(b) < tolerance:
            return [(-np.inf, np.inf)] if c <= 0 else []
        root = -c / b
        return [(-np.inf, root)] if b > 0 else [(root, np.inf)]
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return [(-np.inf, np.inf)] if a < 0 else []
    roots = sorted([(-b - np.sqrt(discriminant)) / (2 * a),
                    (-b + np.sqrt(discriminant)) / (2 * a)])
    return [(roots[0], roots[1])] if a > 0 else [(-np.inf, roots[0]), (roots[1], np.inf)]


def ar_confidence_set(frame, alpha=0.05, treatment="cut", outcome="sold_30d"):
    x = sm.add_constant(frame[["x", "recommendation"]]).to_numpy()
    y, d = frame[outcome].to_numpy(), frame[treatment].to_numpy()
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("The instrument/controls are rank deficient.")
    projection = np.linalg.solve(x.T @ x, x.T)
    by, bd = projection @ y, projection @ d
    ey, ed = y - x @ by, d - x @ bd
    weights = projection[-1] ** 2 * len(x) / (len(x) - x.shape[1])
    vy, vd, vyd = np.sum(weights * ey**2), np.sum(weights * ed**2), np.sum(weights * ey * ed)
    critical = chi2.ppf(1 - alpha, 1)
    return quadratic_set(
        bd[-1] ** 2 - critical * vd,
        -2 * by[-1] * bd[-1] + 2 * critical * vyd,
        by[-1] ** 2 - critical * vy,
    )


def format_intervals(intervals):
    if not intervals:
        return "empty confidence set"
    return " U ".join(
        f"[{'-inf' if np.isneginf(lo) else f'{lo:.4f}'}, "
        f"{'inf' if np.isposinf(hi) else f'{hi:.4f}'}]" for lo, hi in intervals
    )


def estimate_iv(frame, treatment="cut", reviewer=False, outcome="sold_30d"):
    model = IV2SLS.from_formula(
        f"{outcome} ~ 1 + x + [{treatment} ~ recommendation]", frame
    )
    options = {"cov_type": "clustered", "clusters": frame.reviewer} if reviewer else {
        "cov_type": "robust"
    }
    result = model.fit(**options)
    diagnostics = result.first_stage.diagnostics.loc[treatment]
    controls = sm.add_constant(frame[["x", "recommendation"]])
    first = sm.OLS(frame[treatment], controls).fit(cov_type="HC1")
    reduced = sm.OLS(frame[outcome], controls).fit(cov_type="HC1")
    interval = result.conf_int().loc[treatment]
    ar = ar_confidence_set(frame, treatment=treatment, outcome=outcome) if not reviewer else None
    return {
        "estimate": float(result.params[treatment]),
        "lower": float(interval.iloc[0]), "upper": float(interval.iloc[1]),
        "first_stage": float(first.params["recommendation"]),
        "reduced_form": float(reduced.params["recommendation"]),
        "partial_r_squared": float(diagnostics["partial.rsquared"]),
        "first_stage_statistic": float(diagnostics["f.stat"]),
        "first_stage_distribution": diagnostics["f.dist"],
        "ar_confidence_set": format_intervals(ar) if ar is not None else
            "Not computed: this AR implementation assumes independent observations.",
        "ar_unbounded": any(not np.isfinite([lo, hi]).all() for lo, hi in ar) if ar else False,
        "inference_note": "Reviewer-clustered 2SLS; only 16 clusters, interpret cautiously."
        if reviewer else "Heteroskedasticity-robust; exclusion is a design assumption, not a test result.",
    }


def instrument_balance(frame):
    """An observed pre-assignment check, not an exclusion or hidden-balance test."""
    controls = sm.add_constant(frame[["recommendation"]])
    result = sm.OLS(frame.x, controls).fit(cov_type="HC1")
    return {
        "covariate": "x",
        "slope_per_instrument_unit": float(result.params["recommendation"]),
        "p_value": float(result.pvalues["recommendation"]),
        "interpretation": "Observed balance cannot certify exclusion or balance of hidden demand.",
    }


def exclusion_sensitivity(frame, direct_effects=(-0.02, -0.01, 0, 0.01, 0.02)):
    """Conditional inference under an assumed additive direct Z -> Y effect."""
    if not np.isin(frame.recommendation, [0, 1]).all():
        raise ValueError("This sensitivity grid uses binary recommendation units.")
    rows = []
    for direct in direct_effects:
        if not np.isfinite(direct) or abs(direct) > 1:
            raise ValueError("A direct probability effect must be finite and in [-1, 1].")
        adjusted = frame.assign(adjusted_outcome=frame.sold_30d - direct * frame.recommendation)
        fit = estimate_iv(adjusted, outcome="adjusted_outcome")
        rows.append({
            "assumed_direct_effect": float(direct), "adjusted_effect": fit["estimate"],
            "lower": fit["lower"], "upper": fit["upper"],
            "ar_confidence_set": fit["ar_confidence_set"],
            "ar_unbounded": fit["ar_unbounded"],
        })
    return pd.DataFrame(rows)


def continuous_offer_sample(n=8000, seed=42):
    """Constant-slope bounded LPM fixture: known probability effect per log offer."""
    rng = np.random.default_rng(seed)
    x, u = rng.uniform(-1, 1, (2, n))
    z = rng.choice([-1.0, 0.0, 1.0], n)
    log_offer = 0.015 * z - 0.015 * u + rng.uniform(-0.004, 0.004, n)
    probability = 0.45 + 3 * log_offer + 0.04 * x + 0.16 * u
    if not np.all((probability > 0) & (probability < 1)):
        raise ValueError("The continuous-IV fixture violated its bounded probability contract.")
    return pd.DataFrame({
        "x": x, "recommendation": z, "log_offer": log_offer,
        "accepted_14d": rng.binomial(1, probability),
    }), {"oracle_slope": 3.0, "units": "probability per unit log(offer / baseline)",
         "warning": "Constant-slope educational model; heterogeneous continuous IV has a weighted target."}


def belief_information_example(seed=42, n=5000):
    """Information ITT, not a valid instrument for price or an identified mediation effect."""
    rng = np.random.default_rng(seed)
    z = rng.binomial(1, 0.5, n)
    prior = rng.normal(0.02, 0.01, n)
    posterior = prior + 0.02 * z
    q = expit(-0.3 - 12 * posterior)
    y = rng.binomial(1, q)
    return {"information_itt": float(y[z == 1].mean() - y[z == 0].mean()),
            "oracle_information_itt": float(np.mean(
                expit(-0.3 - 12 * (prior + 0.02)) - expit(-0.3 - 12 * prior))),
            "interpretation": "Effect of information assignment, not a price elasticity."}

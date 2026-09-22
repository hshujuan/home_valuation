"""Binary repricing estimands and supported randomized multi-arm response models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from .features import (
    CAUSAL_NUMERIC, action_features, mature_rows, preprocessing, resale_features,
)
from .simulation import ACTIONS


def probability_model(numeric):
    return make_pipeline(preprocessing(numeric),
                         LogisticRegression(C=10, max_iter=1500, random_state=42))


def summarize_scores(scores, name):
    scores = np.asarray(scores, dtype=float)
    if len(scores) < 3 or not np.isfinite(scores).all():
        raise ValueError("At least three finite influence-score observations are required.")
    estimate = float(scores.mean())
    se = float(scores.std(ddof=1) / np.sqrt(len(scores)))
    return {"method": name, "estimate": estimate, "se": se,
            "lower": estimate - 1.96 * se, "upper": estimate + 1.96 * se, "n": len(scores)}


def difference_in_means(frame):
    frame = mature_rows(frame)
    treated = frame.loc[frame.cut.eq(1), "sold_30d"]
    control = frame.loc[frame.cut.eq(0), "sold_30d"]
    if min(len(treated), len(control)) < 2:
        raise ValueError("Both price-action groups need at least two mature outcomes.")
    estimate = float(treated.mean() - control.mean())
    se = float(np.sqrt(treated.var() / len(treated) + control.var() / len(control)))
    return {"method": "difference_in_means", "estimate": estimate, "se": se,
            "lower": estimate - 1.96 * se, "upper": estimate + 1.96 * se, "n": len(frame)}


def doubly_robust(frame, seed=42, randomized=False):
    frame = resale_features(mature_rows(frame)).reset_index(drop=True)
    d, y = frame.cut.to_numpy(), frame.sold_30d.to_numpy()
    if len(np.unique(d)) != 2 or min(np.bincount(d)) < 6:
        raise ValueError("Cross-fitting requires both treatment arms with adequate observations.")
    mu0, mu1, propensity = (np.empty(len(frame)) for _ in range(3))
    splitter = StratifiedKFold(3, shuffle=True, random_state=seed)
    for train, test in splitter.split(frame, d):
        fit, hold = frame.iloc[train], frame.iloc[test]
        if randomized:
            propensity[test] = 0.5
        else:
            pmodel = probability_model(CAUSAL_NUMERIC).fit(fit, fit.cut)
            propensity[test] = pmodel.predict_proba(hold)[:, 1]
        for arm, target in [(0, mu0), (1, mu1)]:
            subset = fit.loc[fit.cut.eq(arm)]
            if subset.sold_30d.nunique() < 2:
                raise ValueError("An outcome-training fold has only one outcome class.")
            outcome = probability_model(CAUSAL_NUMERIC).fit(subset, subset.sold_30d)
            target[test] = outcome.predict_proba(hold)[:, 1]
    if np.any((propensity <= 0) | (propensity >= 1)):
        raise ValueError("Numerically zero/one propensity: the AIPW estimate is not supported.")
    scores = mu1 - mu0 + d * (y - mu1) / propensity - (1 - d) * (y - mu0) / (1 - propensity)
    result = summarize_scores(scores, "cross_fitted_AIPW")
    result.update({
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
        "poor_overlap_fraction": float(np.mean((propensity < 0.05) | (propensity > 0.95))),
        "assumption": "random assignment" if randomized else "no unmeasured confounding",
    })
    return result


def balance_table(frame):
    frame = resale_features(frame)
    rows = []
    for column in CAUSAL_NUMERIC:
        a, b = frame.loc[frame.cut.eq(1), column], frame.loc[frame.cut.eq(0), column]
        pooled = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"feature": column, "standardized_mean_difference":
                     float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan})
    return pd.DataFrame(rows)


RESPONSE_NUMERIC = CAUSAL_NUMERIC + ["log_action", "action_condition"]


@dataclass
class ResponseModel:
    estimator: object

    def predict(self, frame, action):
        if not np.isclose(action, ACTIONS).any():
            raise ValueError("Unsupported action. Use the randomized grid: -6%, -3%, 0%, +3%.")
        return self.estimator.predict_proba(action_features(frame, action))[:, 1]

    def matrix(self, frame):
        return np.column_stack([self.predict(frame, action) for action in ACTIONS])


def fit_response(frame):
    if not frame.design.eq("multiarm").all():
        raise ValueError("Policy response requires the randomized multi-arm study.")
    frame = mature_rows(frame)
    if frame.sold_30d.nunique() != 2:
        raise ValueError("Response training requires both completed-sale outcomes.")
    if not all(np.isclose(frame.action, a).sum() >= 10 for a in ACTIONS):
        raise ValueError("Insufficient randomized support for one or more candidate actions.")
    model = probability_model(RESPONSE_NUMERIC)
    model.fit(action_features(frame), frame.sold_30d)
    return ResponseModel(model)


def fit_associational_response(frame):
    if not frame.design.eq("confounded_multiarm").all():
        raise ValueError("This comparison model expects the confounded multi-arm fixture.")
    model = probability_model(RESPONSE_NUMERIC)
    model.fit(action_features(frame), frame.sold_30d)
    return ResponseModel(model)

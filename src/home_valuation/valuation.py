"""Prediction and split-conformal intervals; neither is a causal price model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from .features import VALUE_NUMERIC, add_comps, preprocessing


@dataclass
class ValuationModel:
    estimator: object
    radius: float
    name: str
    smearing: float = 1.0

    def predict(self, frame):
        return self.estimator.predict(frame) * self.smearing

    def interval(self, frame):
        prediction = self.predict(frame)
        return np.maximum(0, prediction - self.radius), prediction + self.radius


def error_metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "mean_absolute_relative_error": float(np.mean(np.abs(prediction / y - 1))),
        "p90_absolute_relative_error": float(np.quantile(np.abs(prediction / y - 1), 0.9)),
        "bias": float(np.mean(prediction - y)),
    }


@threadpool_limits.wrap(limits=1)
def fit_valuation(history):
    frame = add_comps(history, history).sort_values("market_day")
    cuts = np.quantile(frame.market_day.unique(), [0.6, 0.7, 0.8])
    train = frame.loc[frame.market_day < cuts[0]]
    tune = frame.loc[(frame.market_day >= cuts[0]) & (frame.market_day < cuts[1])]
    calibration = frame.loc[(frame.market_day >= cuts[1]) & (frame.market_day < cuts[2])]
    test = frame.loc[frame.market_day >= cuts[2]]
    models = {}
    smearing = {}
    scores = []
    for name, regressor in [
        ("hedonic", Ridge(alpha=10)),
        ("boosted", HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=15,
                                                  l2_regularization=10, random_state=42)),
    ]:
        model = TransformedTargetRegressor(
            regressor=make_pipeline(preprocessing(VALUE_NUMERIC), regressor),
            func=np.log, inverse_func=np.exp,
        )
        model.fit(train, train.sale_price)
        models[name] = model
        smearing[name] = float(np.mean(train.sale_price / model.predict(train)))
        scores.append({"model": name, "split": "tuning",
                       **error_metrics(tune.sale_price, model.predict(tune) * smearing[name])})
        scores.append({"model": name, "split": "test",
                       **error_metrics(test.sale_price, model.predict(test) * smearing[name])})
    best = min((r for r in scores if r["split"] == "tuning"), key=lambda r: r["mae"])["model"]
    estimator = models[best]
    residual = np.abs(calibration.sale_price - estimator.predict(calibration) * smearing[best])
    level = min(1, np.ceil((len(residual) + 1) * 0.9) / len(residual))
    radius = float(np.quantile(residual, level, method="higher"))
    fitted = ValuationModel(estimator, radius, best, smearing[best])
    prediction = fitted.predict(test)
    lo, hi = fitted.interval(test)
    predictions = test[["home_id", "market", "market_day", "sale_price", "condition"]].copy()
    predictions["prediction"] = prediction
    predictions["lower"] = lo
    predictions["upper"] = hi
    geo_train = train.loc[train.market.ne("Metro-D")]
    geo_test = test.loc[test.market.eq("Metro-D")]
    geo_model = TransformedTargetRegressor(
        regressor=make_pipeline(preprocessing(VALUE_NUMERIC), Ridge(alpha=10)),
        func=np.log, inverse_func=np.exp,
    ).fit(geo_train, geo_train.sale_price)
    geo_test = add_comps(geo_test, history.loc[history.market.ne("Metro-D")])
    geo_smearing = float(np.mean(geo_train.sale_price / geo_model.predict(geo_train)))
    scores.append({"model": "held_out_geography", "split": "test",
                   **error_metrics(geo_test.sale_price, geo_model.predict(geo_test) * geo_smearing)})
    summary = {
        "selected_model": best, "nominal_coverage": 0.9,
        "temporal_test_coverage": float(np.mean((test.sale_price >= lo) & (test.sale_price <= hi))),
        "interval_width": 2 * radius,
        "train_last_day": int(train.market_day.max()),
        "test_first_day": int(test.market_day.min()),
        "warning": "Split conformal coverage requires exchangeability; temporal/market shift can break it.",
    }
    return fitted, pd.DataFrame(scores), predictions, summary

"""A single reproducible pipeline shared by CLI, examples, and the local demo."""

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .causal import (
    balance_table, difference_in_means, doubly_robust, fit_associational_response, fit_response,
)
from .evaluation import (
    bootstrap_curve, calibration_table, condition_reporting_ablation, iv_monte_carlo,
    probability_metrics, stress_curves, valuation_slices,
)
from .features import add_comps
from .instruments import (
    belief_information_example, continuous_offer_sample, estimate_iv,
    exclusion_sensitivity, instrument_balance,
)
from .optimization import (
    choose_actions, completion_frontier, example_curve, off_policy_value,
    paired_policy_difference, policy_comparison, profit_matrices,
)
from .seller import acquisition_curves, fit_seller, homeowner_profiles
from .simulation import (
    ACTIONS, Config, iv_sample, market_sales, resale_log, resale_population,
    reviewer_sample, seller_offers, toy_cut_probability,
)
from .valuation import fit_valuation
from .visualization import save_figures


@dataclass
class Study:
    config: Config
    tables: dict
    metadata: dict
    models: dict
    truth: dict


@threadpool_limits.wrap(limits=1)
def run_study(config=Config(), bootstrap_reps=0, mc_reps=0):
    history, market_truth = market_sales(config.n_market, config.seed)
    avm, metrics, predictions, avm_summary = fit_valuation(history)
    population, resale_truth = resale_population(
        config.n_resale, config.seed + 1, history, avm)
    tables = {"market_sales": history, "valuation_metrics": metrics,
              "valuation_predictions": predictions, "valuation_slices": valuation_slices(predictions)}
    truth_tables = {"market": market_truth, "resale": resale_truth}
    effects, balances = [], []
    for design in ["randomized", "observed", "hidden"]:
        frame, truth = resale_log(population, resale_truth, config.seed + 2, design)
        tables[f"resale_{design}"] = frame
        for result in [difference_in_means(frame),
                       doubly_robust(frame, config.seed, randomized=design == "randomized")]:
            result.update(design=design, oracle_ate=float((truth.q_cut3 - truth.q_hold).mean()))
            effects.append(result)
        balances.append(balance_table(frame).assign(design=design))
    tables["causal_effects"] = pd.DataFrame(effects)
    tables["balance"] = pd.concat(balances, ignore_index=True)
    multiarm, multi_truth = resale_log(population, resale_truth, config.seed + 3, "multiarm")
    cutoff = multiarm.market_day.quantile(0.7)
    train = multiarm.loc[multiarm.market_day < cutoff].copy()
    test = multiarm.loc[multiarm.market_day >= cutoff].copy()
    test_truth = multi_truth.loc[test.index].copy()
    response = fit_response(train)
    predicted_q = np.empty(len(test))
    for action in ACTIONS:
        mask = np.isclose(test.action, action)
        predicted_q[mask] = response.predict(test.loc[mask], action)
    tables["resale_multiarm"] = multiarm
    tables["response_calibration"] = calibration_table(test.sold_30d, predicted_q)
    tables["policy_comparison"], policies, expected = policy_comparison(test, test_truth, response)
    observational, _ = resale_log(
        population, resale_truth, config.seed + 4, "confounded_multiarm")
    predictive = fit_associational_response(observational.loc[observational.market_day < cutoff])
    pred_expected = profit_matrices(test, predictive.matrix(test))
    pred_choices = choose_actions(pred_expected["contribution"], pred_expected["prices"])
    from .simulation import oracle_resale_q
    oracle_q = np.column_stack([oracle_resale_q(test, test_truth, a) for a in ACTIONS])
    oracle_reward = profit_matrices(test, oracle_q, terminal_values=test_truth.true_value)
    selected = np.arange(len(test)), pred_choices
    tables["policy_comparison"] = pd.concat([
        tables["policy_comparison"],
        pd.DataFrame([{
            "policy": "predictive_only_confounded",
            "oracle_sale_probability": oracle_q[selected].mean(),
            "oracle_gross_profit": oracle_reward["gross"][selected].mean(),
            "oracle_contribution": oracle_reward["contribution"][selected].mean(),
            "oracle_regret": (oracle_reward["contribution"].max(axis=1) -
                              oracle_reward["contribution"][selected]).mean(),
            "mean_action": ACTIONS[pred_choices].mean(),
        }]),
    ], ignore_index=True)
    policies["predictive_only_confounded"] = pred_choices
    tables["policy_offline_evaluation"] = pd.DataFrame([
        {"policy": name, **off_policy_value(test, choice, expected)}
        for name, choice in policies.items() if not name.startswith("oracle")
    ])
    tables["policy_paired_comparisons"] = pd.DataFrame([
        {"policy": name, **paired_policy_difference(test, choice, policies["hold"], expected)}
        for name, choice in policies.items() if not name.startswith("oracle")
    ])
    tables["portfolio_frontier"] = completion_frontier(test, test_truth, expected)
    profile = test.iloc[[len(test) // 2]].copy()
    tables["profile_curve"] = example_curve(profile, response)
    tables["stress_curves"] = stress_curves(profile, response)
    if bootstrap_reps:
        tables["bootstrap_curve"] = bootstrap_curve(train, profile, bootstrap_reps, config.seed + 200)
    sellers, seller_truth = seller_offers(config.n_sellers, config.seed + 5, history, avm)
    seller_model, seller_metrics = fit_seller(sellers)
    tables["seller_offers"] = sellers
    truth_tables["seller"] = seller_truth
    tables["acquisition_curves"] = acquisition_curves(sellers, seller_truth, seller_model)
    tables["homeowner_profiles"] = homeowner_profiles()
    iv_results, iv_checks, iv_sensitivity = [], [], []
    for scenario in ["strong", "weak", "invalid", "nonrandom"]:
        frame, target = iv_sample(config.n_iv, config.seed + 6, scenario)
        tables[f"iv_{scenario}"] = frame
        iv_results.append({"scenario": scenario, **estimate_iv(frame), **target})
        iv_checks.append({"scenario": scenario, **instrument_balance(frame)})
        iv_sensitivity.append(exclusion_sensitivity(frame).assign(scenario=scenario))
    for direct in [False, True]:
        reviewer = reviewer_sample(config.n_iv, config.seed + 7, direct)
        iv_results.append({"scenario": "reviewer_direct_path" if direct else "reviewer_past_score",
                           **estimate_iv(reviewer, reviewer=True)})
    tables["iv_results"] = pd.DataFrame(iv_results)
    tables["iv_assignment_checks"] = pd.DataFrame(iv_checks)
    tables["iv_exclusion_sensitivity"] = pd.concat(iv_sensitivity, ignore_index=True)
    continuous, continuous_truth = continuous_offer_sample(config.n_iv, config.seed + 8)
    tables["continuous_iv"] = continuous
    continuous_result = estimate_iv(
        continuous, treatment="log_offer", outcome="accepted_14d")
    continuous_result.update(continuous_truth)
    continuous_result["local_3pct_approximation"] = continuous_result["estimate"] * np.log(0.97)
    continuous_result["extrapolation_warning"] = (
        "Assignment shifts log offer by only +/-0.015. A -3% move is not a randomized finite contrast."
    )
    if mc_reps:
        tables["iv_monte_carlo"] = iv_monte_carlo(mc_reps, min(4000, config.n_iv), config.seed + 1000)
        mc = tables["iv_monte_carlo"]
        coverage = float(mc.covered.mean())
        tables["iv_monte_carlo_summary"] = pd.DataFrame([{
            "repetitions": mc_reps, "mean_error": float(mc.error.mean()),
            "rmse": float(np.sqrt(np.mean(mc.error**2))), "wald_coverage": coverage,
            "coverage_monte_carlo_se": float(np.sqrt(coverage * (1 - coverage) / mc_reps)),
        }])
    prepared_test = add_comps(history.loc[history.home_id.isin(predictions.home_id)], history)
    tables["condition_ablation"] = condition_reporting_ablation(history, avm, prepared_test)
    metadata = {
        "config": config.to_dict(), "schema_version": 1,
        "valuation": avm_summary, "seller_metrics": seller_metrics,
        "response_metrics": probability_metrics(test.sold_30d, predicted_q),
        "continuous_iv": continuous_result,
        "belief_information": belief_information_example(config.seed),
        "toy_q0": 0.4, "toy_q_after_3pct_cut": toy_cut_probability(0.4, -12),
        "profile_home_id": int(profile.home_id.iloc[0]),
        "response_train_n": len(train), "response_test_n": len(test),
        "bootstrap_repetitions": bootstrap_reps, "monte_carlo_repetitions": mc_reps,
        "limitations": [
            "Synthetic results are not Opendoor causal estimates.",
            "Thirty-day completion and day-45 terminal sale are different endpoints.",
            "Independent-home experiments omit equilibrium/interference effects.",
            "Bootstrap intervals condition on a frozen AVM and do not cover all model risk.",
            "Stress-robust actions maximize the worst of named economic scenarios, not a statistical guarantee.",
            "Information ITT and continuous-IV slopes are not the binary repricing ATE.",
            "Portfolio completion constraints use fitted probabilities, not realized-volume guarantees.",
            "Paired policy intervals are pointwise, not multiple-comparison-adjusted.",
            "IV sensitivity intervals condition on each assumed direct effect; they do not identify it.",
        ],
    }
    return Study(config, tables, metadata, {"avm": avm, "response": response,
                 "seller": seller_model, "profile": profile, "response_train": train}, truth_tables)


def markdown_table(frame):
    def display(value):
        if pd.isna(value):
            return "not applicable"
        if isinstance(value, (float, np.floating)):
            return f"{value:.5f}"
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(display(v) for v in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_results(study, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    sections = [
        "# Reproducible synthetic results", "",
        "Generated by the local pipeline, not copied from an external research report.",
        "All dollar amounts and response effects are synthetic. Probabilities are fractions unless labeled otherwise.",
        "", "Configuration: `" + json.dumps(study.config.to_dict(), sort_keys=True) + "`", "",
    ]
    for name in ["valuation_metrics", "valuation_slices", "causal_effects", "iv_results", "profile_curve",
                 "acquisition_curves", "policy_comparison", "policy_offline_evaluation",
                 "condition_ablation", "stress_curves", "bootstrap_curve", "iv_monte_carlo_summary"]:
        if name not in study.tables:
            continue
        sections += ["## " + name.replace("_", " ").title(), "",
                     markdown_table(study.tables[name]), ""]
    for name in ["iv_assignment_checks", "iv_exclusion_sensitivity",
                 "policy_paired_comparisons", "portfolio_frontier"]:
        sections += ["## " + name.replace("_", " ").title(), "",
                     markdown_table(study.tables[name]), ""]
    sections += ["## Uncertainty and provenance", "",
                 "Metadata records split sizes, estimator units, seed, and all study limitations.",
                 "An IV confidence interval is meaningful only under the stated identification assumptions.",
                 "Weak-IV confidence sets can be unbounded. No causal validation claim follows from first-stage strength alone.",
                 "", "```json", json.dumps(study.metadata, indent=2, allow_nan=False), "```", ""]
    (destination / "reference_results.md").write_text("\n".join(sections), encoding="utf-8")
    save_figures(study.tables, destination / "figures")


def save_study(study, output, public_report=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"config": study.config.to_dict(), "schema_version": 1, "artifacts": {}}
    for name, frame in study.tables.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    private = output / "evaluator_truth"
    private.mkdir(exist_ok=True)
    for name, frame in study.truth.items():
        frame.to_csv(private / f"{name}.csv", index=False)
    manifest["packages"] = {name: version(name) for name in
                            ["numpy", "pandas", "scikit-learn", "linearmodels"]}
    source = Path(__file__).parent
    manifest["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(source.glob("*.py"))}
    (output / "metadata.json").write_text(
        json.dumps(study.metadata, indent=2, allow_nan=False), encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    write_results(study, output)
    if public_report:
        write_results(study, Path(__file__).resolve().parents[2] / "docs")

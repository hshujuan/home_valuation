"""Linked fixed-stage policies, continuation learning, and sequential DR evaluation."""

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from .causal import probability_model, summarize_scores
from .features import preprocessing
from .pipeline import markdown_table
from .simulation import market_sales
from .two_sided_simulation import (
    QUOTE_COST, SELLING_RATE, STAGE_ACTIONS, TwoSidedConfig, action_indices,
    feasible_actions, generate_cohort,
)
from .valuation import fit_valuation


BUY_FEATURES = [
    "log_sqft", "condition", "age", "belief_ratio", "initial_offer_ratio",
    "attachment_score", "convenience_score", "engagement_score",
    "repair_ratio", "margin_ratio", "max_escalation",
]
SELL_FEATURES = ["log_sqft", "condition", "age", "buyer_signal", "list_ratio"]
ACTION_FEATURES = ["action_pp", "action_squared", "action_condition", "action_state"]


def require_maturity(frame, stage):
    column = {"buy": "buy_mature_at", "sell": "sell_mature_at",
              "lifecycle": "reward_mature_at"}[stage]
    end, observed = pd.to_datetime(frame[column]), pd.to_datetime(frame.observed_through)
    if end.isna().any() or observed.isna().any() or (observed < end).any():
        raise ValueError(f"Fully observed {stage} outcomes are required; missing is not negative.")


def stage_features(frame, stage, action):
    if stage not in {"buy", "sell"} or not frame[f"{stage}_eligible"].all():
        raise ValueError("Use the eligible population for the specified stage.")
    decision = pd.to_datetime(frame[f"{stage}_decision_date"])
    available = pd.to_datetime(frame[f"{stage}_features_at"])
    if available.isna().any() or decision.isna().any() or (available >= decision).any():
        raise ValueError("Stage features must be available before the intervention.")
    if not np.isfinite(frame.avm_value).all() or (frame.avm_value <= 0).any():
        raise ValueError("Finite positive as-of AVM values are required.")
    result = frame.copy()
    result["log_sqft"] = np.log(result.sqft / 1800)
    if stage == "buy":
        result["belief_ratio"] = result.belief_before_offer / result.avm_value
        result["repair_ratio"] = result.repair_forecast / result.avm_value
        result["margin_ratio"] = result.margin_forecast / result.avm_value
        state = result.belief_ratio - result.initial_offer_ratio
    else:
        result["list_ratio"] = result.base_list_price / result.avm_value
        state = result.buyer_signal
    delivered = np.broadcast_to(np.asarray(action), len(frame))
    action_indices(delivered)
    result["action_pp"] = delivered * 100
    result["action_squared"] = result.action_pp**2
    result["action_condition"] = result.action_pp * result.condition
    result["action_state"] = result.action_pp * state
    columns = (BUY_FEATURES if stage == "buy" else SELL_FEATURES) + ACTION_FEATURES + ["market"]
    return result[columns]


@dataclass
class StageResponse:
    estimator: object
    stage: str
    closing_rate: float = 1.0

    def matrix(self, frame):
        return np.column_stack([
            self.estimator.predict_proba(stage_features(frame, self.stage, a))[:, 1]
            * self.closing_rate for a in STAGE_ACTIONS
        ])


def fit_stage_response(frame, stage):
    require_maturity(frame, stage)
    actions = action_indices(frame[f"{stage}_action"])
    if min(np.bincount(actions, minlength=4)) < 10:
        raise ValueError("Every stage action needs at least ten randomized training observations.")
    outcome = "accepted_14d" if stage == "buy" else "sold_30d"
    endpoints = ([outcome, "completed_acquisition_35d"] if stage == "buy" else [outcome])
    if not frame[endpoints].isin([0, 1]).all().all() or frame[outcome].nunique() != 2:
        raise ValueError("Stage training needs observed outcomes in both classes.")
    if stage == "buy" and (frame.completed_acquisition_35d > frame.accepted_14d).any():
        raise ValueError("Completed acquisition cannot precede acceptance.")
    features = BUY_FEATURES if stage == "buy" else SELL_FEATURES
    estimator = probability_model(features + ACTION_FEATURES).fit(
        stage_features(frame, stage, frame[f"{stage}_action"]), frame[outcome])
    closing = (float(frame.loc[frame.accepted_14d.eq(1), "completed_acquisition_35d"].mean())
               if stage == "buy" else 1.0)
    return StageResponse(estimator, stage, closing)


def resale_values(frame, q, terminal_ratio=0.94):
    if q.shape != (len(frame), 4) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Provide four finite resale probabilities per eligible home.")
    if not np.isfinite(terminal_ratio) or not 0 < terminal_ratio <= 1.3:
        raise ValueError("The terminal-value ratio must be in (0, 1.3].")
    prices = frame.base_list_price.to_numpy()[:, None] * (1 - STAGE_ACTIONS)
    revenue = q * 0.99 * prices + (1 - q) * terminal_ratio * frame.avm_value.to_numpy()[:, None]
    days = 18 * q + 45 * (1 - q)
    return (1 - SELLING_RATE) * revenue - frame.daily_carrying_cost.to_numpy()[:, None] * days


def supported_best(values, feasible):
    if values.shape != feasible.shape or not np.isfinite(values).all():
        raise ValueError("Finite objectives must match the feasible action matrix.")
    if not feasible.any(axis=1).all():
        raise ValueError("No feasible stage action; abstain rather than extrapolate.")
    return np.where(feasible, values, -np.inf).argmax(axis=1)


def select_escalation(values, feasible, min_gain=150.0, tolerance=150.0):
    if min_gain < 0 or tolerance < 0 or not np.isfinite([min_gain, tolerance]).all():
        raise ValueError("Gain hurdle and near-optimal tolerance must be finite and nonnegative.")
    if not feasible[:, 0].all():
        raise ValueError("Holding the existing offer must be supported.")
    best = supported_best(values, feasible)
    maximum = values[np.arange(len(values)), best]
    gains = values - values[:, [0]]
    acceptable = feasible & (values >= maximum[:, None] - tolerance) & (gains >= min_gain)
    acceptable[:, 0] = False
    chosen = np.where(acceptable.any(axis=1), acceptable.argmax(axis=1), 0)
    actual_gain = gains[np.arange(len(values)), chosen]
    return chosen, actual_gain


def stage_scores(frame, choices, predictions, outcomes, stage):
    feasible = feasible_actions(frame, stage)
    choices = np.asarray(choices)
    n = len(frame)
    if choices.shape != (n,) or not np.isin(choices, np.arange(4)).all():
        raise ValueError("One supported action index is required per row.")
    choices = choices.astype(int)
    if not feasible[np.arange(n), choices].all():
        raise ValueError("The target policy violates conditional experimental support.")
    assigned = action_indices(frame[f"{stage}_action"])
    if not feasible[np.arange(n), assigned].all():
        raise ValueError("A logged action violates its predecision feasibility constraint.")
    p = frame[f"{stage}_assignment_probability"].to_numpy()
    if not np.isfinite(p).all() or not np.allclose(p, 1 / feasible.sum(axis=1)):
        raise ValueError("Known within-feasible-set randomization probabilities are required.")
    outcomes = np.asarray(outcomes)
    if (predictions.shape != (n, 4) or not np.isfinite(predictions).all()
            or outcomes.shape != (n,) or not np.isfinite(outcomes).all()):
        raise ValueError("Finite observed outcomes and four predictions per row are required.")
    weight = (assigned == choices) / p
    score = predictions[np.arange(n), choices] + weight * (
        outcomes - predictions[np.arange(n), assigned])
    return score, weight


def continuation_rewards(frame, response, sell_rule, terminal_ratio=0.94):
    """DR downstream value for all buy prospects, including genuine nonacquisitions."""
    require_maturity(frame, "lifecycle")
    if sell_rule not in {"hold", "optimized"}:
        raise ValueError("Sell rule must be hold or optimized.")
    acquired = frame.acquired.astype(bool)
    closes = pd.to_datetime(frame.resale_close_date)
    if (closes[acquired].isna().any()
            or (closes[acquired] > pd.to_datetime(frame.loc[acquired, "observed_through"])).any()):
        raise ValueError("Acquired inventory needs an observed mature disposition reward.")
    net = frame.realized_disposition_net.copy()
    downstream_weight = pd.Series(1.0, index=frame.index)
    weak = frame.loc[frame.sell_eligible].copy()
    if len(weak):
        require_maturity(weak, "sell")
        values = resale_values(weak, response.matrix(weak), terminal_ratio)
        choices = (supported_best(values, feasible_actions(weak, "sell"))
                   if sell_rule == "optimized" else np.zeros(len(weak), dtype=int))
        scores, weight = stage_scores(
            weak, choices, values, weak.realized_remaining_net.to_numpy(), "sell")
        elapsed = (weak.sell_decision_date - weak.acquisition_close_date).dt.days
        net.loc[weak.index] = scores - weak.daily_carrying_cost.to_numpy() * elapsed.to_numpy()
        downstream_weight.loc[weak.index] = weight
    if (not np.isfinite(net[acquired]).all()
            or not np.isfinite(frame.loc[acquired, ["company_outlay", "repair_cost"]]).all().all()):
        raise ValueError("Observed disposition and repair costs are required for acquired homes.")
    reward = -QUOTE_COST * frame.buy_eligible.astype(float)
    reward.loc[acquired] += (net[acquired] - frame.loc[acquired, "company_outlay"]
                             - frame.loc[acquired, "repair_cost"])
    return reward.to_numpy(), downstream_weight.to_numpy()


@dataclass
class ContinuationModel:
    estimator: object

    def matrix(self, frame):
        return np.column_stack([
            self.estimator.predict(stage_features(frame, "buy", a)) * frame.avm_value.to_numpy()
            for a in STAGE_ACTIONS
        ])


def fit_continuation(frame, rewards):
    require_maturity(frame, "lifecycle")
    if not np.isfinite(rewards).all():
        raise ValueError("Continuation-learning labels must be finite.")
    model = make_pipeline(preprocessing(BUY_FEATURES + ACTION_FEATURES), Ridge(alpha=50))
    model.fit(stage_features(frame, "buy", frame.buy_action),
              rewards / frame.avm_value.to_numpy())
    return ContinuationModel(model)


def stage_effects(frame, truth, response, stage):
    require_maturity(frame, stage)
    common = frame.loc[feasible_actions(frame, stage).all(axis=1)].copy()
    if len(common) < 30:
        raise ValueError("The common-support population is too small for dose contrasts.")
    q = response.matrix(common)
    target = "completed_acquisition_35d" if stage == "buy" else "sold_30d"
    baseline, _ = stage_scores(common, np.zeros(len(common), int), q, common[target], stage)
    aligned_truth = truth.set_index("home_id").loc[common.home_id]
    oracle0 = aligned_truth[f"{stage}_close_q_0"].to_numpy()
    rows = []
    for j, action in enumerate(STAGE_ACTIONS):
        score, _ = stage_scores(common, np.full(len(common), j), q, common[target], stage)
        difference = summarize_scores(score - baseline, "randomized_common_support_AIPW")
        rows.append({
            "stage": stage, "action_magnitude": action, "common_support_n": len(common),
            "fitted_completion": q[:, j].mean(), "aipw_completion": score.mean(),
            "effect_vs_hold": difference["estimate"], "lower": difference["lower"],
            "upper": difference["upper"],
            "oracle_effect_vs_hold": float(
                (aligned_truth[f"{stage}_close_q_{j}"].to_numpy() - oracle0).mean()),
        })
    return pd.DataFrame(rows)


@dataclass
class TwoSidedStudy:
    config: TwoSidedConfig
    tables: dict
    metadata: dict
    models: dict
    truth: dict


@threadpool_limits.wrap(limits=1)
def run_two_sided(config=TwoSidedConfig()):
    history, _ = market_sales(config.n_market, config.seed)
    avm, _, _, avm_summary = fit_valuation(history)
    cohorts, truth = {}, {}
    for offset, (name, size, start) in enumerate([
        ("development", config.n_development, 760),
        ("continuation", config.n_continuation, 1160),
        ("evaluation", config.n_evaluation, 1560),
    ]):
        cohorts[name], truth[name] = generate_cohort(
            size, config.seed + 500 + offset, name, start, history, avm)
    for before, after in [("development", "continuation"), ("continuation", "evaluation")]:
        if cohorts[before].reward_mature_at.max() >= cohorts[after].offer_date.min():
            raise ValueError("Historical lifecycle outcomes must mature before the next cohort.")
    development, continuation, evaluation = (cohorts[name] for name in
                                             ["development", "continuation", "evaluation"])
    buy_response = fit_stage_response(development.loc[development.buy_eligible], "buy")
    sell_response = fit_stage_response(development.loc[development.sell_eligible], "sell")
    learn = continuation.loc[continuation.buy_eligible].copy()
    test = evaluation.loc[evaluation.buy_eligible].copy()
    models, labels = {}, {}
    for rule in ["hold", "optimized"]:
        labels[rule], _ = continuation_rewards(learn, sell_response, rule)
        models[rule] = fit_continuation(learn, labels[rule])
    values = models["optimized"].matrix(test)
    choices, gains = select_escalation(values, feasible_actions(test, "buy"))
    predictions = {"hold": models["hold"].matrix(test), "optimized": values}
    outcomes = {rule: continuation_rewards(test, sell_response, rule) for rule in models}
    policies = [
        ("hold_both", np.zeros(len(test), int), "hold"),
        ("resale_only", np.zeros(len(test), int), "optimized"),
        ("linked_selective", choices, "optimized"),
        ("maximum_feasible_escalation", feasible_actions(test, "buy").sum(axis=1) - 1, "optimized"),
    ]
    policy_rows, cohort_rows, baseline, cohort_baseline = [], [], None, None
    full_outcomes = {rule: continuation_rewards(evaluation, sell_response, rule)[0] for rule in models}
    for name, buy_choices, sell_rule in policies:
        reward, downstream_weight = outcomes[sell_rule]
        scores, weight = stage_scores(test, buy_choices, predictions[sell_rule], reward, "buy")
        if baseline is None:
            baseline = scores
        value = summarize_scores(scores, "sequential_DR")
        difference = summarize_scores(scores - baseline, "paired_sequential_DR")
        joint_weight = weight * downstream_weight
        ess = joint_weight.sum()**2 / np.sum(joint_weight**2) if np.any(joint_weight) else 0
        policy_rows.append({
            "policy": name, "n_eligible_sellers": len(test), "dr_value": value["estimate"],
            "value_lower": value["lower"], "value_upper": value["upper"],
            "difference_vs_hold_both": difference["estimate"],
            "difference_lower": difference["lower"], "difference_upper": difference["upper"],
            "escalation_share": float(np.mean(buy_choices > 0)),
            "path_weight_ess": float(ess),
        })
        full_scores = pd.Series(full_outcomes[sell_rule], index=evaluation.index, copy=True)
        full_scores.loc[test.index] = scores
        if cohort_baseline is None:
            cohort_baseline = full_scores.to_numpy()
        cohort_value = summarize_scores(full_scores, "linked_cohort_DR")
        cohort_difference = summarize_scores(
            full_scores.to_numpy() - cohort_baseline, "paired_linked_cohort_DR")
        cohort_rows.append({
            "policy": name, "n_original_prospects": len(evaluation),
            "dr_value_per_original_prospect": cohort_value["estimate"],
            "value_lower": cohort_value["lower"], "value_upper": cohort_value["upper"],
            "difference_vs_hold_both": cohort_difference["estimate"],
            "difference_lower": cohort_difference["lower"],
            "difference_upper": cohort_difference["upper"],
        })
    details = test[["home_id", "initial_offer", "margin_forecast", "max_escalation"]].copy()
    details["chosen_escalation"] = STAGE_ACTIONS[choices]
    details["predicted_incremental_value"] = gains
    details["predicted_joint_value_chosen"] = values[np.arange(len(test)), choices]
    details["predicted_joint_value_hold"] = values[:, 0]
    details["resale_policy_link_at_hold"] = values[:, 0] - predictions["hold"][:, 0]
    allowed = feasible_actions(test, "buy")
    best = supported_best(values, allowed)
    details["best_feasible_predicted_gain"] = values[np.arange(len(test)), best] - values[:, 0]
    candidates = pd.DataFrame({
        "home_id": np.repeat(test.home_id.to_numpy(), 4),
        "action_magnitude": np.tile(STAGE_ACTIONS, len(test)),
        "feasible": allowed.ravel(),
        "gross_offer": (test.initial_offer.to_numpy()[:, None] * (1 + STAGE_ACTIONS)).ravel(),
        "fitted_completion": np.where(allowed, buy_response.matrix(test), np.nan).ravel(),
        "joint_value_hold_resale": np.where(allowed, predictions["hold"], np.nan).ravel(),
        "joint_value_optimized_resale": np.where(allowed, values, np.nan).ravel(),
        "incremental_value_optimized_resale": np.where(
            allowed, values - values[:, [0]], np.nan).ravel(),
    })
    common = candidates.loc[candidates.home_id.isin(test.loc[allowed.all(axis=1), "home_id"])]
    buy_curve = common.groupby("action_magnitude").agg(
        common_support_n=("home_id", "size"),
        mean_gross_offer=("gross_offer", "mean"),
        fitted_completion=("fitted_completion", "mean"),
        joint_value_hold_resale=("joint_value_hold_resale", "mean"),
        joint_value_optimized_resale=("joint_value_optimized_resale", "mean"),
        incremental_value_optimized_resale=("incremental_value_optimized_resale", "mean"),
    ).reset_index()
    learned_labels = learn[["home_id", "buy_action"]].copy()
    learned_labels["hold_resale_reward_label"] = labels["hold"]
    learned_labels["optimized_resale_reward_label"] = labels["optimized"]
    tables = {**cohorts, "policy_evaluation": pd.DataFrame(policy_rows),
              "cohort_policy_evaluation": pd.DataFrame(cohort_rows),
              "buy_decisions": details, "buy_candidate_values": candidates,
              "buy_value_curve": buy_curve, "continuation_learning": learned_labels}
    tables["stage_effects"] = pd.concat([
        stage_effects(test, truth["evaluation"], buy_response, "buy"),
        stage_effects(evaluation.loc[evaluation.sell_eligible], truth["evaluation"],
                      sell_response, "sell"),
    ], ignore_index=True)
    tables["cohort_funnel"] = pd.DataFrame([{
        "cohort": name, "prospects": len(frame),
        "initial_acceptances": int(frame.initial_accepted_7d.sum()),
        "escalation_eligible": int(frame.buy_eligible.sum()),
        "stage_acceptances_14d": int(frame.accepted_14d.sum()),
        "stage_acquisitions_35d": int(frame.completed_acquisition_35d.sum()),
        "all_acquisitions": int(frame.acquired.sum()),
        "early_resales": int(frame.early_sold.sum()),
        "markdown_eligible": int(frame.sell_eligible.sum()),
        "post_markdown_closes_30d": int(frame.loc[frame.sell_eligible, "sold_30d"].sum()),
        "first_offer": str(frame.offer_date.min().date()),
        "last_reward_mature": str(frame.reward_mature_at.max().date()),
    } for name, frame in cohorts.items()])
    tables["buy_action_mix"] = details.groupby("chosen_escalation").agg(
        homes=("home_id", "size"), mean_predicted_gain=("predicted_incremental_value", "mean")
    ).reset_index()
    metadata = {
        "config": config.to_dict(), "valuation": avm_summary,
        "stage_actions": STAGE_ACTIONS.tolist(), "min_predicted_gain": 150,
        "near_optimal_tolerance": 150,
        "buy_closing_given_acceptance_estimate": buy_response.closing_rate,
        "value_target": "Lifecycle contribution per engaged initial nonacceptor under fixed joint policies.",
        "cohort_value_target": "Modeled contribution per original quoted prospect; initial offer policy "
                              "and eligibility are fixed, initial marketing/quote costs are excluded.",
        "randomization": "Full delivery, uniform within the predecision feasible set; no IV required.",
        "limitations": [
            "Synthetic assumptions and independent homes; no externally validated company policy.",
            "Initial offers and noneligible paths are fixed; this is not an optimized initial-offer policy.",
            "One escalation and one markdown, not a repeated-decision Bellman solver.",
            "The joint acquisition objective learns completed-acquisition-weighted continuation rewards, "
            "not conversion multiplied by an unselected mean margin.",
            "Both nuisance-learning cohorts mature before final evaluation; no online or future labels.",
            "Stage dose contrasts use their own common-support populations; do not compare as one ATE.",
            "Sell-stage contrasts condition on acquisition under the historical logging policy.",
            "Policy intervals are pointwise and conditional on trained models; no multiplicity guarantee.",
            "Acquisition closing after acceptance is action-independent in this simulator.",
            "The $150 hurdle concerns the chosen predicted gain, not a guaranteed realized gain.",
        ],
    }
    return TwoSidedStudy(config, tables, metadata,
                         {"buy_response": buy_response, "sell_response": sell_response,
                          "continuation": models}, truth)


def write_two_sided_report(study, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Linked two-sided extension: generated results", "",
        "This is an additive synthetic experiment, not a replacement for the original study.",
        "Probabilities are fractions. Policy values are USD per escalation-eligible prospect, "
        "except the explicitly labeled all-original-prospects table.",
        "A negative lower bound or weak policy ranking is retained, not tuned away.", "",
        "Configuration: `" + json.dumps(study.config.to_dict(), sort_keys=True) + "`", "",
    ]
    for name in ["cohort_funnel", "stage_effects", "buy_value_curve", "policy_evaluation",
                 "cohort_policy_evaluation", "buy_action_mix"]:
        parts += ["## " + name.replace("_", " ").title(), "",
                  markdown_table(study.tables[name]), ""]
    parts += [
        "## Interpretation", "",
        "Stage contrasts are restricted to homes eligible for every action at that stage.",
        "The joint policy score instead covers all engaged initial nonacceptors, including those "
        "whose only feasible acquisition action is hold and those who never acquire.",
        "The resale-only comparison keeps acquisition escalation at zero; linked-selective also "
        "may change escalation using the learned joint continuation objective. "
        "Inspect the action mix: a policy is allowed to choose hold for everyone.", "",
        "The cohort-policy table uses all original prospects, including initial acceptors "
        "and nonengaged sellers. Its denominator differs from the stage-eligible table. "
        "Initial offer/eligibility rules are fixed and initial marketing/quote costs are excluded.", "",
        "Only past cohorts supply model parameters. Oracle probabilities are used for the "
        "stage-effect diagnostic column, never for policy selection.", "",
        "![Linked stages and held-out policy comparisons](figures/two_sided_extension.png)", "",
        "See the repository's `docs/data_dictionary.md` and "
        "`docs/design_implementation_summary.md` for timing, support, accounting, "
        "and inference assumptions. Stage-score and continuation-reward equations "
        "are implemented in `src/home_valuation/two_sided.py`.", "",
    ]
    (destination / "two_sided_results.md").write_text("\n".join(parts), encoding="utf-8")
    directory = destination / "figures"
    directory.mkdir(exist_ok=True)
    effects = study.tables["stage_effects"]
    policies = study.tables["policy_evaluation"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for axis, stage, title in zip(axes[:2], ["buy", "sell"],
                                  ["Acquisition completion by day 35", "Resale completion by day 30"]):
        rows = effects.loc[effects.stage.eq(stage)]
        axis.plot(rows.action_magnitude * 100, rows.fitted_completion * 100,
                  marker="o", label="Fitted")
        axis.plot(rows.action_magnitude * 100, rows.aipw_completion * 100,
                  marker="s", label="Held-out AIPW")
        axis.set(title=title, xlabel="Increase offer (%)" if stage == "buy" else "Cut list price (%)",
                 ylabel="Completion probability (%)")
        axis.legend()
    y = np.arange(len(policies))
    center = policies.difference_vs_hold_both.to_numpy()
    axes[2].errorbar(center, y, xerr=np.vstack([
        center - policies.difference_lower, policies.difference_upper - center]), fmt="o")
    axes[2].axvline(0, color="gray", linestyle="--")
    axes[2].set(yticks=y, yticklabels=policies.policy, xlabel="Contribution difference ($ / prospect)",
                title="Paired sequential DR vs hold both")
    fig.tight_layout()
    fig.savefig(directory / "two_sided_extension.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_two_sided(study, output, public_report=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, table in study.tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    private = output / "evaluator_truth"
    private.mkdir(exist_ok=True)
    truth_hashes = {}
    for name, table in study.truth.items():
        path = private / f"{name}.csv"
        table.to_csv(path, index=False)
        truth_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "metadata.json").write_text(json.dumps(study.metadata, indent=2), encoding="utf-8")
    manifest = {
        "config": study.config.to_dict(), "artifacts": hashes,
        "schema_version": 1, "evaluator_truth_sha256": truth_hashes,
        "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn"]},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_two_sided_report(study, output)
    if public_report:
        write_two_sided_report(study, Path(__file__).resolve().parents[2] / "docs")

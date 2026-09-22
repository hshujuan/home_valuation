from pathlib import Path
import re

import numpy as np
import plotly.express as px
import streamlit as st

from home_valuation.optimization import Economics, choose_actions, example_curve
from home_valuation.pipeline import run_study
from home_valuation.simulation import Config
from home_valuation.two_sided import run_two_sided
from home_valuation.two_sided_simulation import TwoSidedConfig
from home_valuation.visualization import profit_figure, response_figure

st.set_page_config(page_title="Home valuation and causal pricing lab", layout="wide")
st.title("Two-sided home valuation and causal pricing lab")
st.caption(
    "Synthetic visual demo: valuation uncertainty, homeowner offer response, "
    "buyer repricing response, and lifecycle profit decisions."
)
st.caption("Synthetic educational data. Not Opendoor's model, data, or pricing advice.")


@st.cache_resource(show_spinner="Generating and fitting the reproducible synthetic study...")
def study_for(seed, size):
    return run_study(Config(seed=seed, n_market=size, n_sellers=size,
                            n_resale=size, n_iv=size))


@st.cache_resource(show_spinner="Learning the separate linked two-sided policies...")
def two_sided_for(seed, size):
    cohort_size = max(3600, size * 2)
    return run_two_sided(TwoSidedConfig(
        seed=seed, n_market=size, n_development=cohort_size,
        n_continuation=cohort_size, n_evaluation=cohort_size))


seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42)
size = st.sidebar.selectbox("Observations per table", [1000, 3000, 6000], index=1)
section = st.sidebar.radio("Section", [
    "Overview and EDA", "Valuation", "Homeowner decisions", "Causal repricing",
    "Instrumental variables", "Profit optimization", "Two-sided extension",
    "Pipeline and assumptions",
])
study = study_for(int(seed), size)
tables, metadata = study.tables, study.metadata
st.sidebar.info("Resale price cut: buyer conversion. Acquisition offer: homeowner acceptance.")

if section == "Overview and EDA":
    st.subheader("1. Understand the data before fitting")
    a, b, c = st.columns(3)
    a.metric("Synthetic market transactions", len(tables["market_sales"]))
    b.metric("Eligible repricing decisions", len(tables["resale_randomized"]))
    c.metric("Observed 30-day sale rate", f"{tables['resale_randomized'].sold_30d.mean():.1%}")
    table_name = st.selectbox("Dataset", [
        "market_sales", "seller_offers", "resale_randomized", "resale_multiarm",
    ])
    frame = tables[table_name]
    st.dataframe(frame.head(40), width="stretch")
    st.dataframe(frame.isna().mean().rename("missing_fraction").to_frame(), width="stretch")
    st.plotly_chart(px.histogram(tables["market_sales"], x="sale_price", color="market",
                                title="Synthetic sale-price distribution"), width="stretch")
    st.warning("Future outcomes are included for learning and evaluation, never as model inputs.")

elif section == "Valuation":
    st.subheader("2. Predict value, quantify error, respect time")
    st.json(metadata["valuation"])
    st.dataframe(tables["valuation_metrics"], width="stretch")
    st.dataframe(tables["valuation_slices"], width="stretch")
    st.plotly_chart(px.scatter(
        tables["valuation_predictions"], x="sale_price", y="prediction", color="market",
        title="Out-of-time valuation predictions"), width="stretch")
    st.dataframe(tables["condition_ablation"], width="stretch")
    st.info("Condition ablation holds the trained model fixed and substitutes optimistic self-reports. "
            "Conformal coverage is measured, not guaranteed under market/time shift.")

elif section == "Homeowner decisions":
    st.subheader("3. Asset value is not the homeowner's reservation utility")
    st.markdown(
        "Separate **market belief**, **verified condition**, **outside options**, and "
        "**net proceeds**. A higher acquisition offer usually increases seller acceptance; "
        "that is the opposite direction from a higher resale listing price."
    )
    st.plotly_chart(px.line(
        tables["homeowner_profiles"], x="offer", y="acceptance_probability",
        color="profile", markers=True, title="Stipulated seller profiles; identical $500,000 asset"),
        width="stretch")
    st.dataframe(tables["acquisition_curves"], width="stretch")
    st.warning("Selection-aware profit and true acquired value above use evaluator-only simulation "
               "truth. They are diagnostics, not deployable estimates from an AVM alone.")
    st.json(metadata["belief_information"])
    st.caption("The information scenario identifies assignment ITT, not the causal effect of price.")

elif section == "Causal repricing":
    st.subheader("4. A 3% list-price cut among homes still unsold")
    st.markdown("**Treatment:** reduce current list price to 97%. **Control:** hold price. "
                "**Outcome:** completed resale in the following 30 days.")
    effects = tables["causal_effects"].copy()
    effects["effect_pp"] = effects.estimate * 100
    effects["interval_pp"] = effects.se * 1.96 * 100
    effects["label"] = effects.design + " / " + effects.method
    st.plotly_chart(px.scatter(
        effects, x="effect_pp", y="label", error_x="interval_pp",
        title="Estimates with 95% intervals, in percentage points"), width="stretch")
    st.dataframe(effects[["design", "method", "estimate", "lower", "upper", "oracle_ate"]],
                 width="stretch")
    st.dataframe(tables["balance"], width="stretch")
    st.info("Observed: the confounder is measured. Hidden: the same confounder is only weakly proxied. "
            "AIPW/cross-fitting cannot manufacture identification.")
    st.plotly_chart(px.line(tables["response_calibration"], x="predicted_probability",
                           y="observed_rate", markers=True,
                           title="Held-out randomized response calibration"), width="stretch")

elif section == "Instrumental variables":
    st.subheader("5. Instrument design before estimator choice")
    scenario = st.selectbox("IV scenario", tables["iv_results"].scenario.tolist())
    row = tables["iv_results"].loc[tables["iv_results"].scenario.eq(scenario)]
    st.dataframe(row.T.rename(columns={row.index[0]: "value"}), width="stretch")
    st.warning("First-stage strength does not prove exclusion. IV estimates are local/design-specific, "
               "not a global price-response curve.")
    if scenario == "weak":
        st.error("Weak-instrument example: inspect the Anderson-Rubin confidence set, "
                 "which may be unbounded or disjoint.")
    if scenario in {"invalid", "nonrandom", "reviewer_direct_path"}:
        st.error("The simulator deliberately violates identification in this scenario. "
                 "Do not interpret the coefficient as a valid causal estimate.")
    if scenario in set(tables["iv_exclusion_sensitivity"].scenario):
        st.dataframe(tables["iv_assignment_checks"].query("scenario == @scenario"), width="stretch")
        st.dataframe(tables["iv_exclusion_sensitivity"].query("scenario == @scenario"),
                     width="stretch")
        st.caption("Sensitivity uses an assumed additive direct effect of recommendation on "
                   "completion, in probability units. Each interval conditions on that assumption. "
                   "Observed covariate balance does not prove instrument validity.")
    with st.expander("Continuous acquisition-offer companion"):
        st.json(metadata["continuous_iv"])
        st.markdown("This fixture's outcome is **seller acceptance**, not resale completion. "
                    "A local derivative and a finite 3% policy effect are different estimands.")

elif section == "Profit optimization":
    st.subheader("6. Conversion is not profit")
    source = tables["resale_multiarm"]
    home = st.number_input("Home row", 0, len(source) - 1,
                           int(study.models["profile"].index[0]))
    objective = st.selectbox("Objective", ["Expected contribution", "Expected gross profit"])
    a, b, c = st.columns(3)
    terminal = a.slider("Terminal proceeds / value", 0.80, 1.05, 0.94, 0.01)
    holding = b.slider("Holding-cost multiplier", 0.5, 5.0, 1.0, 0.5)
    floor = c.slider("Minimum price / current price", 0.80, 1.15, 0.80, 0.01)
    profile = source.iloc[[int(home)]]
    curve = example_curve(profile, study.models["response"],
                          Economics(terminal_ratio=terminal, holding_multiplier=holding))
    st.plotly_chart(response_figure(curve), width="stretch")
    st.plotly_chart(profit_figure(curve), width="stretch")
    st.dataframe(curve, width="stretch")
    try:
        choice = choose_actions(
            curve["expected_contribution" if objective == "Expected contribution"
                  else "expected_gross_profit"].to_numpy()[None, :],
            curve.list_price.to_numpy()[None, :],
            [float(profile.base_list_price.iloc[0]) * floor],
        )[0]
    except ValueError as error:
        st.error(str(error))
    else:
        st.success(f"Best supported action for this illustrative objective: "
                   f"{curve.action.iloc[choice]:+.0%}; price ${curve.list_price.iloc[choice]:,.0f}")
    if np.any(np.diff(curve.sale_probability) > 0):
        st.warning("The fitted curve is not monotone here. Investigate sampling/model error; "
                   "the simulator's structural curve is monotone, but fitted estimates need not be.")
    st.download_button("Download this synthetic scenario", curve.to_csv(index=False),
                       "synthetic_price_scenario.csv", "text/csv")
    st.caption("Only randomized actions -6%, -3%, 0%, +3% are eligible. Stress controls change "
               "payoff assumptions, not the estimated causal response. Unsold inventory is sold "
               "under a stipulated day-45 continuation policy, not valued at zero.")
    st.dataframe(tables["policy_comparison"], width="stretch")
    st.dataframe(tables["policy_offline_evaluation"], width="stretch")
    st.dataframe(tables["policy_paired_comparisons"], width="stretch")
    st.caption("Paired intervals compare policies on the same holdout homes against hold. "
               "They are pointwise and do not control selection across many comparisons.")
    st.subheader("Contribution versus required portfolio completion")
    frontier = tables["portfolio_frontier"]
    st.plotly_chart(px.line(frontier, x="predicted_completion", y="predicted_contribution",
                           markers=True, hover_data=["policy", "shadow_price_per_expected_completion"],
                           title="Supported-price lotteries under an expected completion constraint"),
                    width="stretch")
    st.dataframe(frontier, width="stretch")
    st.caption("The optimizer may randomize between supported prices; it never substitutes their "
               "average price. Targets use fitted probabilities, not guaranteed realized volume. "
               "Oracle columns expose model error for simulation evaluation only.")
    st.caption("This frontier uses the current run's holdout homes and reference economics. "
               "The individual-home controls above do not change this portfolio calculation.")
    st.caption("The stress-robust policy maximizes the worst of five prespecified economic "
               "scenarios. It is not a confidence bound or a guarantee against all market risks.")
    st.dataframe(tables["stress_curves"], width="stretch")

elif section == "Two-sided extension":
    st.subheader("7. Link selective acquisition escalation to resale markdown")
    st.info("Additive experiment: the original 3% resale study is unchanged. "
            "Raise the acquisition offer by 0/1/2/3%; cut resale price by 0/1/2/3%. "
            "Actions are randomized within predecision limits, with full delivery; no IV is needed.")
    linked = two_sided_for(int(seed), size)
    data = linked.tables
    st.caption(f"This app run uses {linked.config.n_market:,} historical transactions and "
               f"{linked.config.n_development:,} prospects in each of three chronological cohorts. "
               "Earlier lifecycle outcomes mature before the next cohort begins.")
    st.dataframe(data["cohort_funnel"], width="stretch")
    effects = data["stage_effects"].copy()
    effects["effect_pp"] = 100 * effects.effect_vs_hold
    effects["interval_pp"] = 100 * (effects.upper - effects.effect_vs_hold)
    st.plotly_chart(px.line(
        effects, x="action_magnitude", y="effect_pp", color="stage",
        error_y="interval_pp", markers=True,
        title="Held-out AIPW completion lift vs hold; different stage populations",
        labels={"action_magnitude": "Proportional action magnitude", "effect_pp": "Lift (pp)"}),
        width="stretch")
    st.caption("Buy: completed acquisition within 35 days, not acceptance alone. "
               "Sell: completed resale within 30 days after the markdown checkpoint. "
               "Each curve uses its own all-actions-feasible population, not a shared ATE.")
    st.dataframe(effects, width="stretch")
    st.subheader("Downstream policy feeds the acquisition objective")
    st.dataframe(data["buy_value_curve"], width="stretch")
    st.caption("Same common-support prospects at every action. Joint dollar rewards are learned "
               "on all eligible prospects, including genuine nonacquisitions; they are not "
               "completion probability times an unconditional average acquired-home margin.")
    candidates = data["buy_candidate_values"]
    home_id = st.selectbox("Linked prospect", data["buy_decisions"].home_id.head(100).tolist())
    st.dataframe(candidates.loc[candidates.home_id.eq(home_id)], width="stretch")
    st.caption("Unsupported candidate predictions are left empty, never recommended. "
               "Individual fitted values are model estimates, not identified personal effects.")
    st.subheader("Frozen joint policies: held-out lifecycle contribution")
    policies = data["policy_evaluation"].copy()
    policies["interval"] = policies.difference_upper - policies.difference_vs_hold_both
    st.plotly_chart(px.scatter(
        policies, x="difference_vs_hold_both", y="policy", error_x="interval",
        title="Paired sequential DR difference vs hold both ($ / eligible prospect)"),
        width="stretch")
    st.dataframe(data["policy_evaluation"], width="stretch")
    st.dataframe(data["cohort_policy_evaluation"], width="stretch")
    st.caption("The second table uses all original prospects, including initial acceptors. "
               "Its denominator differs; initial offers and initial marketing/quote costs "
               "are not optimized or fully modeled.")
    st.dataframe(data["buy_action_mix"], width="stretch")
    st.warning("Synthetic independent-home assumptions. Pointwise intervals condition on frozen "
               "models, not policy-search or market-wide uncertainty. A $150 chosen predicted-gain "
               "hurdle is not guaranteed realized profit. Holding every offer is a valid result.")
    st.download_button("Download linked policy comparisons",
                       data["policy_evaluation"].to_csv(index=False),
                       "synthetic_two_sided_policy_evaluation.csv", "text/csv")
    with st.expander("Linked configuration and limitations"):
        st.json(linked.metadata)

else:
    st.subheader("8. Reproduce and inspect assumptions")
    st.json(metadata)
    st.code(
        "python -m home_valuation pipeline --seed 42 --write-report\n"
        "python examples\\01_eda.py\n"
        "python -m pytest", language="powershell")
    st.info("The app uses the selected sample size with no bootstrap/Monte Carlo loop. "
            "The reference CLI run records its own configuration and uncertainty artifacts.")
    report = Path(__file__).resolve().parents[1] / "docs" / "analysis_report.md"
    if report.exists():
        with st.expander("Read the analysis report"):
            text = report.read_text(encoding="utf-8")
            st.download_button("Download analysis Markdown", text, "analysis_report.md", "text/markdown")
            chunks = re.split(r"!\[([^\]]*)\]\((figures/[^)]+)\)", text)
            for index in range(0, len(chunks), 3):
                st.markdown(chunks[index])
                if index + 2 < len(chunks):
                    st.image(str(report.parent / chunks[index + 2]), caption=chunks[index + 1])
            st.caption("Relative documentation links are intended for the files on disk or GitHub.")
    else:
        st.warning("The analysis report is not present at docs\\analysis_report.md.")

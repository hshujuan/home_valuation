from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st

from home_valuation.optimization import Economics, choose_actions, example_curve
from home_valuation.pipeline import run_study
from home_valuation.simulation import Config
from home_valuation.two_sided import run_two_sided
from home_valuation.two_sided_simulation import TwoSidedConfig
from home_valuation.visualization import profit_figure, response_figure

st.set_page_config(
    page_title="Home valuation and causal pricing lab",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "ink": "#17352f",
    "teal": "#147d70",
    "mint": "#83b9a9",
    "sand": "#e4a85f",
    "coral": "#d76f51",
    "blue": "#547d9b",
}
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    COLORS["teal"], COLORS["sand"], COLORS["blue"], COLORS["coral"], COLORS["mint"]
]

st.markdown(
    """
    <style>
    :root {
        --ink: #17352f;
        --muted: #63736e;
        --teal: #147d70;
        --teal-dark: #0d5f56;
        --mint: #dcece6;
        --cream: #f8f5ee;
        --sand: #e4a85f;
        --border: #dbe3df;
        --white: #ffffff;
        --shadow: 0 12px 32px rgba(23, 53, 47, 0.08);
    }

    html, body, [class*="css"] {
        font-family: "Aptos", "Segoe UI", sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(circle at 88% 3%, rgba(228, 168, 95, 0.13), transparent 24rem),
            linear-gradient(180deg, #fbfaf7 0%, #f6f8f6 34rem, #f8faf9 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(251, 250, 247, 0.82);
        backdrop-filter: blur(12px);
    }

    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1440px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        font-family: "Aptos Display", "Segoe UI", sans-serif;
        color: var(--ink);
        letter-spacing: -0.035em;
    }

    h2 {
        margin-top: 1.2rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid rgba(23, 53, 47, 0.09);
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .hero {
        position: relative;
        overflow: hidden;
        min-height: 280px;
        margin-bottom: 1.8rem;
        padding: 3.1rem 3.3rem;
        border: 1px solid rgba(20, 125, 112, 0.14);
        border-radius: 28px;
        background:
            linear-gradient(112deg, rgba(10, 80, 72, 0.98), rgba(20, 125, 112, 0.91)),
            #147d70;
        box-shadow: 0 22px 55px rgba(18, 91, 82, 0.2);
    }

    .hero::before {
        content: "";
        position: absolute;
        width: 380px;
        height: 380px;
        right: -85px;
        top: -145px;
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 48% 52% 58% 42%;
        transform: rotate(18deg);
    }

    .hero::after {
        content: "⌂";
        position: absolute;
        right: 4.5rem;
        bottom: 1.1rem;
        color: rgba(255, 255, 255, 0.08);
        font-size: 12rem;
        line-height: 1;
    }

    .hero-kicker {
        position: relative;
        z-index: 1;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.38rem 0.72rem;
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 999px;
        color: #e8f4f0;
        background: rgba(255, 255, 255, 0.09);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        max-width: 820px;
        margin: 1.25rem 0 0.85rem;
        color: white;
        font-size: clamp(2.35rem, 4vw, 4.25rem);
        line-height: 1.02;
        letter-spacing: -0.055em;
    }

    .hero p {
        position: relative;
        z-index: 1;
        max-width: 760px;
        margin: 0;
        color: rgba(255, 255, 255, 0.78);
        font-size: 1.06rem;
        line-height: 1.65;
    }

    .hero-tags {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1.45rem;
    }

    .hero-tag {
        padding: 0.45rem 0.78rem;
        border-radius: 9px;
        color: #f7fbfa;
        background: rgba(255, 255, 255, 0.1);
        font-size: 0.82rem;
        font-weight: 600;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #dce4e0;
        background: linear-gradient(180deg, #f1f7f4 0%, #f8f6ef 100%);
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.5rem;
    }

    .sidebar-brand {
        margin: 0.2rem 0 1.3rem;
        padding: 1rem 1rem 1.05rem;
        border-radius: 16px;
        color: white;
        background: var(--ink);
        box-shadow: var(--shadow);
    }

    .sidebar-brand strong {
        display: block;
        font-family: "Aptos Display", "Segoe UI", sans-serif;
        font-size: 1rem;
        letter-spacing: -0.02em;
    }

    .sidebar-brand span {
        display: block;
        margin-top: 0.2rem;
        color: #a9c9c0;
        font-size: 0.76rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        margin-bottom: 0.22rem;
        padding: 0.5rem 0.65rem;
        border-radius: 9px;
        transition: background 140ms ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(20, 125, 112, 0.08);
    }

    [data-testid="stMetric"] {
        min-height: 118px;
        padding: 1.15rem 1.25rem;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.84);
        box-shadow: var(--shadow);
    }

    [data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        font-family: "Aptos Display", "Segoe UI", sans-serif;
        color: var(--teal-dark);
        letter-spacing: -0.04em;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"],
    [data-testid="stJson"] {
        overflow: hidden;
        padding: 0.65rem;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 7px 22px rgba(23, 53, 47, 0.055);
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
        border-width: 1px;
        box-shadow: 0 6px 18px rgba(23, 53, 47, 0.04);
    }

    [data-testid="stExpander"] {
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.7);
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 2.75rem;
        border: 1px solid var(--teal);
        border-radius: 10px;
        color: white;
        background: var(--teal);
        font-weight: 700;
        box-shadow: 0 7px 16px rgba(20, 125, 112, 0.15);
        transition: transform 140ms ease, box-shadow 140ms ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--teal-dark);
        color: white;
        background: var(--teal-dark);
        transform: translateY(-1px);
        box-shadow: 0 10px 21px rgba(20, 125, 112, 0.22);
    }

    .stSelectbox [data-baseweb="select"] > div,
    .stNumberInput [data-baseweb="input"] > div,
    .stTextInput [data-baseweb="input"] > div {
        border-color: var(--border);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.8);
    }

    [data-testid="stSlider"] [role="slider"] {
        background: var(--teal);
    }

    code {
        border-radius: 6px;
        color: var(--teal-dark);
        background: var(--mint);
    }

    @media (max-width: 800px) {
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 1rem;
        }

        .hero {
            min-height: auto;
            padding: 2rem 1.5rem;
            border-radius: 20px;
        }

        .hero h1 {
            font-size: 2.4rem;
        }

        .hero::after {
            display: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Synthetic decision lab</div>
      <h1>Price homes with more confidence.</h1>
      <p>
        Explore how valuation uncertainty, homeowner response, causal repricing,
        and lifecycle economics work together across an iBuyer marketplace.
      </p>
      <div class="hero-tags">
        <span class="hero-tag">Valuation</span>
        <span class="hero-tag">Causal inference</span>
        <span class="hero-tag">Profit optimization</span>
        <span class="hero-tag">Two-sided policy</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)
st.caption("Synthetic educational data. Not Opendoor's model, data, or pricing advice.")

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
      <strong>Home value lab</strong>
      <span>Interactive pricing workspace</span>
    </div>
    """,
    unsafe_allow_html=True,
)


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
    # Transposing mixes estimates, flags, and notes in one column; cast for Arrow display.
    st.dataframe(row.T.rename(columns={row.index[0]: "value"}).astype("string"),
                 width="stretch")
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
    summary = Path(__file__).resolve().parents[1] / "docs" / "design_implementation_summary.md"
    if summary.exists():
        with st.expander("Read the technical summary"):
            text = summary.read_text(encoding="utf-8")
            st.download_button("Download summary Markdown", text,
                               "design_implementation_summary.md", "text/markdown")
            st.markdown(text)
            st.caption("Relative documentation links are intended for the files on disk or GitHub.")
    else:
        st.warning("The technical summary is not present at docs\\design_implementation_summary.md.")

# Interactive demo guide

This guide explains every section of the local Streamlit demo in
[`app/streamlit_app.py`](../app/streamlit_app.py): what it shows, which
controls it offers, which generated tables it reads, and how to interpret it
without overstating what the synthetic study establishes.

Launch it from the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Open **http://127.0.0.1:8501**. The checked-in `.streamlit\config.toml` binds
the server to localhost and disables usage telemetry. Stop it with `Ctrl+C`.

All data is generated at runtime. The app never reads private data, external
services, or the `outputs\` directory, and it never publishes anything.

## How the app is wired

The app is a presentation layer only. It calls the same package entrypoints as
the command line and does not fit separate dashboard-only models.

| App concern | Backing package call |
|---|---|
| Baseline study | `run_study(Config(...))` in [`pipeline.py`](../src/home_valuation/pipeline.py) |
| Linked two-sided study | `run_two_sided(TwoSidedConfig(...))` in [`two_sided.py`](../src/home_valuation/two_sided.py) |
| Price/profit curve | `example_curve()` and `choose_actions()` in [`optimization.py`](../src/home_valuation/optimization.py) |
| Response and profit charts | `response_figure()` and `profit_figure()` in [`visualization.py`](../src/home_valuation/visualization.py) |

Both studies are cached with `st.cache_resource`, keyed by seed and sample
size. Changing either control refits the study; the first run after a change
is therefore slow, and repeat views are fast.

### Sidebar controls

| Control | Effect |
|---|---|
| **Random seed** | Seeds both studies. Different seeds give different synthetic estimates; the documented narrative numbers use seed 42. |
| **Observations per table** | Sets rows per baseline table (1,000 / 3,000 / 6,000; default 3,000). The linked study uses `max(3600, 2 x size)` prospects per cohort. |
| **Section** | Selects one of the eight sections below. |

### Why app numbers differ from the reference run

The app runs with the selected sample size and **no bootstrap or Monte Carlo
repetitions**. The reference CLI run uses default table sizes with resampling
enabled. The app therefore shows the same estimators and decision logic, but
not the exact figures in [`reference_results.md`](reference_results.md) or
[`two_sided_results.md`](two_sided_results.md).

## 1. Overview and EDA

**Purpose.** Inspect the generated data before trusting any model.

**Shows.** Counts of market transactions and eligible repricing decisions, the
observed 30-day sale rate, a preview of a selected dataset, its per-column
missing-value fractions, and the synthetic sale-price distribution by market.

**Controls.** A dataset selector for `market_sales`, `seller_offers`,
`resale_randomized`, and `resale_multiarm`.

**How to read it.** Confirm that scale, missingness, and price dispersion are
plausible before interpreting downstream estimates. Condition is missing for a
fraction of market sales by construction, which is why the preprocessing
pipeline imputes and adds a missingness indicator.

**Caution.** Outcome columns are present for learning and evaluation. They are
never supplied as model inputs at decision time.

## 2. Valuation

**Purpose.** Show that an AVM is a forecasting model evaluated out of time, not
a pricing decision.

**Shows.** The valuation summary metadata (selected model, nominal and realized
interval coverage, interval width, and train/test day boundaries),
`valuation_metrics` across tuning and test splits plus a held-out geography,
`valuation_slices` performance breakdowns, a predicted-versus-actual scatter,
and the `condition_ablation` table.

**How to read it.** Model selection uses the tuning period, the conformal
radius is calibrated on a later period, and reported errors come from the
latest period. Compare the hedonic and boosted models on the same future
holdout: extra flexibility does not automatically forecast better.

**Caution.** The ablation holds the trained model fixed and substitutes
optimistic homeowner-reported condition, quantifying input-quality risk.
Split-conformal coverage assumes exchangeability and can degrade under
temporal or geographic shift, so the measured coverage is reported rather than
assumed.

## 3. Homeowner decisions

**Purpose.** Separate the asset's value from the homeowner's willingness to
sell.

**Shows.** Stipulated seller profiles on an identical $500,000 asset, the
`acquisition_curves` table across offer ratios, and the belief-information
metadata example.

**How to read it.** Three homeowners facing the same asset value accept at
different offers because reservation utility reflects attachment, convenience,
and outside options. In the curves, compare `selection_aware_profit` with
`independence_approximation`, and `joint_slope_per_1pp_offer_ratio` with
`factorized_slope_per_1pp_offer_ratio`: multiplying average acceptance by an
average margin ignores that raising the offer also changes **which** homes are
acquired.

**Direction.** A higher acquisition offer raises seller acceptance. This is
the opposite direction from a higher resale list price, which reduces buyer
completion.

**Caution.** The selection-aware and true-acquired-value columns use
evaluator-only simulator truth. They are diagnostics, not quantities an AVM
alone could deliver. The information scenario identifies an assignment ITT,
not the causal effect of price.

## 4. Causal repricing

**Purpose.** Demonstrate the baseline experiment: a 3% list-price cut versus
holding price, among homes still unsold at the decision, with completed resale
within 30 days as the outcome.

**Shows.** Estimates with 95% intervals for each assignment design and method,
the `causal_effects` table including the oracle ATE, the covariate `balance`
table, and held-out calibration of the fitted response.

**How to read it.** Three designs appear deliberately:

- **randomized** — assignment is independent of the hidden demand signal;
- **observed** — the confounder is measured and adjustable; and
- **hidden** — the same confounder is only weakly proxied.

Compare each estimate against the `oracle_ate` column. The naive observational
difference can have the opposite sign from the randomized effect, because
operational markdowns concentrate on harder-to-sell inventory.

**Caution.** Cross-fitted doubly robust estimation improves precision and
robustness, but it cannot manufacture identification when a confounder is
unmeasured.

## 5. Instrumental variables

**Purpose.** Show that instrument **design** matters more than estimator
choice.

**Shows.** Per-scenario IV results, assignment balance checks, exclusion
sensitivity, and a continuous acquisition-offer companion in an expander.

**Controls.** A scenario selector covering `strong`, `weak`, `invalid`,
`nonrandom`, and the two reviewer fixtures.

**How to read it.** Each scenario is a fixture with a known failure mode:

| Scenario | Deliberate property |
|---|---|
| `strong` | Valid instrument with a strong first stage |
| `weak` | Valid but barely relevant; inspect the Anderson-Rubin set, which may be unbounded or disjoint |
| `invalid` | Exclusion restriction violated by a direct path |
| `nonrandom` | Instrument assignment depends on the unobserved confounder |
| `reviewer_past_score` / `reviewer_direct_path` | Leave-one-out style reviewer instrument, with and without a direct effect |

The app raises an explicit error banner for scenarios that violate
identification, so a coefficient shown there must not be read as causal.

**Caution.** First-stage strength never proves exclusion. Sensitivity
intervals condition on an *assumed* additive direct effect; they do not
identify it. Observed covariate balance is a necessary diagnostic, not proof
of validity. The continuous-offer fixture's outcome is seller acceptance, and
a local derivative is not the same estimand as a finite 3% policy contrast.

## 6. Profit optimization

**Purpose.** Show that conversion is not profit, and that choices must stay
inside the tested action set.

**Shows.** A per-home response curve and profit curve, the numeric curve table,
the recommended supported action, `policy_comparison`,
`policy_offline_evaluation`, `policy_paired_comparisons`, the portfolio
completion frontier, and `stress_curves`.

**Controls.**

| Control | Meaning |
|---|---|
| **Home row** | Selects the illustrated holdout home |
| **Objective** | Expected contribution or expected gross profit |
| **Terminal proceeds / value** | Assumed disposition ratio for still-unsold inventory |
| **Holding-cost multiplier** | Scales daily carrying cost |
| **Minimum price / current price** | A hard price floor on allowed actions |
| **Download** | Exports the displayed synthetic scenario as CSV |

**How to read it.** A deeper cut can raise completion probability while
lowering expected contribution. Raise the price floor above every supported
action to see the app **abstain with an explicit error** instead of inventing
a feasible price. Paired policy comparisons evaluate policies on the same
holdout homes against hold, which is more informative than comparing two
independent intervals.

The portfolio frontier trades fitted expected completion against fitted
contribution using lotteries over supported prices, with shadow prices per
unit of required completion.

**Caution.** Only the randomized actions `-6%`, `-3%`, `0%`, and `+3%` are
eligible; a fractional optimum is a lottery over tested prices, never an
interpolated average price. The stress sliders change payoff assumptions, not
the estimated causal response. The stress-robust policy maximizes the worst of
five prespecified scenarios and is not a confidence bound. The frontier uses
the run's fixed holdout portfolio and reference economics, so the individual
home sliders do not change it. A nonmonotone fitted curve triggers a warning:
the simulator's structural curve is monotone, so that signals sampling or
model error. Unsold inventory is valued under a stipulated day-45 continuation
policy, not at zero.

## 7. Two-sided extension

**Purpose.** Link a selective acquisition escalation to the downstream resale
markdown and evaluate joint policies on lifecycle contribution.

**Shows.** The run's cohort configuration, `cohort_funnel`, held-out AIPW stage
effects, `buy_value_curve`, per-prospect candidate values, frozen policy
comparisons (`policy_evaluation` and `cohort_policy_evaluation`),
`buy_action_mix`, a CSV download, and full linked metadata in an expander.

**Controls.** A linked-prospect selector for per-home candidate values, plus
the shared seed and size controls.

**How to read it.** Actions at both stages are `0/1/2/3%` and are randomized
within predecision feasibility with full delivery, so no instrument is needed.
Three chronological cohorts separate response fitting, downstream-reward
learning, and final evaluation, and each cohort's lifecycle outcomes mature
before the next begins.

Read the two policy tables carefully, because their denominators differ:

- `policy_evaluation` is per **escalation-eligible** prospect; and
- `cohort_policy_evaluation` is per **original** prospect, including initial
  acceptors and nonengaged sellers.

Check `buy_action_mix` before concluding that a policy escalates: holding every
offer is a valid, reported outcome.

**Caution.** Buy-stage effects are completed acquisition within 35 days, not
acceptance alone; sell-stage effects are completed resale within 30 days after
the checkpoint. Each stage curve uses its own all-actions-feasible population,
so the two are not a single shared ATE. Unsupported candidate actions are left
empty and never recommended. The $150 hurdle applies to the chosen action's own
predicted gain, not to guaranteed realized profit. Intervals are pointwise and
condition on frozen models.

## 8. Pipeline and assumptions

**Purpose.** Expose reproduction commands and the run's recorded assumptions.

**Shows.** The complete baseline metadata JSON, including configuration, split
sizes, estimator metadata, and the explicit limitations list, plus the
commands to reproduce the study, run an example, and run the tests.

**Note.** This section also offers an optional inline reader for
`docs\analysis_report.md`. That long-form narrative is not part of this
checkout, so the app displays a warning instead. The maintained equivalents are
the [technical summary](design_implementation_summary.md) and the generated
[reference results](reference_results.md).

## Related documentation

| Document | Purpose |
|---|---|
| [Repository design](architecture.md) | Architecture, module boundaries, study flows, and artifact contracts |
| [Technical design summary](design_implementation_summary.md) | Modeling rationale, estimands, and interpretation |
| [Data dictionary](data_dictionary.md) | Fields, units, timing, and join rules for every generated table |
| [Reference results](reference_results.md) | Seed-42 baseline numbers from the CLI run |
| [Two-sided results](two_sided_results.md) | Seed-42 linked-study funnel, effects, and policy comparisons |

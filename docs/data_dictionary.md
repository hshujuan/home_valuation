# Synthetic data, timing, and accounting contracts

All monetary values are synthetic USD; probabilities are fractions in `[0, 1]`.
In the original experiment, price actions are signed proportional changes:
`-0.03` means multiply the current price by `0.97`. The separate two-sided
extension uses positive magnitudes with stage-specific signs, defined below.
A probability change of `0.08` means **8 percentage points**.

## Files and populations

The original `pipeline` command writes these CSV families beneath its chosen output directory.
Historical market sales feed the AVM; AVM predictions feed the seller and
resale examples. The seller and resale cohorts are separate populations,
not a longitudinal simulation of the exact same acquired homes.
The later `two-sided` command does link the exact same prospects to inventory,
in a different output directory and experiment.

| Table | Row and purpose |
|---|---|
| `market_sales.csv` | One historical completed market transaction, used for AVM development |
| `seller_offers.csv` | One prospect receiving a randomized final acquisition-offer ratio |
| `resale_randomized.csv` | One already-unsold eligible home; randomized hold/3% cut |
| `resale_observed.csv` | Same eligible population; endogenous assignment with its demand confounder revealed |
| `resale_hidden.csv` | Same assignment/outcomes as observed design, but only a noisy demand proxy available |
| `resale_multiarm.csv` | Randomized price actions `-6%, -3%, 0%, +3%`; response fitting and policy evaluation |
| `iv_strong.csv`, `iv_weak.csv`, `iv_invalid.csv`, `iv_nonrandom.csv` | Separate encouragement fixtures, not rows from the main repricing experiment |
| `continuous_iv.csv` | Separate synthetic continuous acquisition-offer IV fixture |
| `valuation_predictions.csv` | Selected AVM predictions and intervals on future test transactions |
| `valuation_metrics.csv`, `valuation_slices.csv` | Aggregate, geography, price-quartile, tail-error, and coverage diagnostics |
| `causal_effects.csv`, `balance.csv` | Binary repricing estimates, intervals, overlap, and balance |
| `response_calibration.csv` | Held-out randomized completion calibration |
| `iv_results.csv` | Binary and reviewer instrument estimates and diagnostics |
| `iv_assignment_checks.csv` | Observed pre-assignment covariate association with recommendation; not a validity test |
| `iv_exclusion_sensitivity.csv` | Binary IV estimates and conditional robust/AR inference under assumed direct recommendation effects |
| `profile_curve.csv`, `stress_curves.csv` | One example home's supported response and economic curves |
| `acquisition_curves.csv`, `homeowner_profiles.csv` | Fitted/oracle selection diagnostics and stipulated seller profiles |
| `policy_comparison.csv` | Oracle evaluation of held-out policies; not production-observable performance |
| `policy_offline_evaluation.csv` | Observable holdout DR policy-value estimates under known randomization |
| `policy_paired_comparisons.csv` | Same-home DR contribution differences versus hold, with pointwise intervals |
| `portfolio_frontier.csv` | Supported-price portfolio lotteries under increasing expected completion constraints |
| `bootstrap_curve.csv` | Optional pointwise percentile bands conditional on the frozen AVM |
| `iv_monte_carlo.csv`, `iv_monte_carlo_summary.csv` | Optional seed-by-seed IV errors and small-sample coverage diagnostic |
| `condition_ablation.csv` | Fixed-model substitution of reported for verified condition |

`home_id` is unique within a generated population, not globally across table
families. Join a population to its corresponding evaluator truth by `home_id`;
do not join unrelated market/seller/resale cohorts just because IDs overlap.
Resale assignment designs intentionally share the same eligible population.

## Common property fields

| Field | Definition and availability |
|---|---|
| `market` | Artificial categorical geography, Metro-A through Metro-D |
| `sqft`, `beds`, `age` | Physical square footage, bedrooms, age in years |
| `condition` | Noisy verified condition score from 0 to 1; missing in some historical rows |
| `reported_condition` | Optimistically noisy homeowner-reported condition; not equivalent to verified condition |
| `market_day` | Integer days from a synthetic time origin; determines chronology |
| `sale_price` | Historical transaction target; never the current row's predictor |
| `avm_value` | Predicted value from the previously fitted AVM, not simulator truth |

Derived AVM features are `log_sqft`, `season`, and
`comp_price_per_sqft`. The last is the median price per square foot of up to
40 **strictly earlier** transactions in the same market. Same-day/future
transactions, including the target's own sale, are excluded. Missing comps
are imputed using training-fitted preprocessing.

## Resale decision and outcome fields

| Field | Definition |
|---|---|
| `listing_date`, `decision_date` | Original listing and repricing decision times |
| `condition_verified_at` | Before listing and before the intervention |
| `days_on_market`, `views_before`, `demand_signal` | Pre-decision inventory age, engagement, and demand measurement |
| `base_list_price` | Price immediately before the new decision |
| `assigned_action`, `action` | Assigned and delivered proportional price changes; full compliance here |
| `override_reason` | Explicit no-override message in the price experiment; IV fixtures separately model noncompliance |
| `list_price` | `base_list_price * (1 + action)` |
| `cut` | Indicator for exactly the 3% cut; only use as the binary treatment in hold/cut designs |
| `assignment_probability` | Known logging probability for randomized designs; unavailable in observational logs |
| `sold_30d` | Completed resale during the next 30 days, not a view, contract, or eventual sale |
| `label_matured`, `observed_through` | Outcome availability; generated rows have mature labels and 60-day observation |
| `contract_date`, `close_date` | Future realized contract and closing times, never predictors at the decision |
| `contract_cancelled_before_terminal` | Illustrative cancellation flag among non-30-day closings; not a full multistate contract model |
| `sale_price_30d` | Realized early proceeds, missing when no sale completes inside 30 days |
| `terminal_sale_price` | Proceeds from the stipulated day-45 liquidation, missing for early sales |
| `realized_revenue`, `realized_gross_profit`, `realized_contribution` | Full synthetic lifecycle reward for offline evaluation |

Eligibility is generated before new treatment assignment. One decision per
home avoids repeated-home standard-error complications in this lab.
`mature_rows` excludes immature or missing outcomes; missing is not zero.
Post-decision dates, prices, outcomes, and realized rewards are never response
model covariates.

## Seller fields

`offer_ratio` is randomized over `0.90, 0.94, 0.98, 1.02` times the AVM.
`belief_before_offer`, `attachment_score`, and `convenience_score` represent
pre-offer information; `belief_elicited_at` and `condition_verified_at` precede
`offer_date`. A belief score is not objectively verified market value.

`accepted_14d` is final-offer acceptance within the synthetic 14-day window.
`completed_acquisition` is a distinct Bernoulli closing event conditional on
acceptance, with stipulated probability `0.96`; no precise acquisition closing
date is modeled. The accepted-only sample must not replace all offered
prospects when fitting the response.

## Financial ledger

These teaching assumptions are not company fees or audited accounting definitions.

| Component | Synthetic rule |
|---|---|
| Gross acquisition offer | `offer_ratio * avm_value` |
| Retained service rate | `0.02` |
| Repair credit | `$6,000`, deducted from the amount paid to acquire the home |
| Company acquisition outlay | `offer * 0.98 - repair_credit` |
| Seller closing cost | `$3,000`, paid away rather than earned by the company |
| Seller net proceeds | `company_outlay - seller_closing_cost` |
| Acquisition selection diagnostic | Expected closing-weighted `true_value * 0.98 - company_outlay - 9000 - 3500 - 6000`, minus `$200` per prospect |
| Additional diagnostic costs | `$9,000` actual repairs, `$3,500` holding, `$6,000` financing/other; fixed teaching costs |
| Resale cohort cost basis | `0.86 * avm_value`, a separate stipulated all-in basis |
| Early resale transaction proceeds | `0.99 * list_price` in expectation, plus mean-one multiplicative noise |
| Resale continuation proceeds | `0.94 * true_value` at day 45 in the generator; planning substitutes the AVM |
| Early closing time | Uniform integer days 10 through 26, conditional mean 18 |
| Selling expense | 2% of realized/expected resale proceeds |
| Holding expense | `0.00010 * avm_value` per day |
| Financing expense | 8% annual rate on basis, simple daily accrual |
| Gross profit in this lab | Lifecycle proceeds minus cost basis, charged once in every state |
| Contribution in this lab | Gross profit minus selling, holding, and financing expense |
| Expected gross-margin rate | Expected gross profit divided by expected lifecycle proceeds |

Repair credits and actual repair expenditure are different entries. Retained
fees already reduce company outlay; do not add them a second time as income.
Do not charge acquisition basis only for homes completing inside 30 days.

The acquisition derivative columns measure a **local change in expected profit
per one percentage-point increase in offer/AVM ratio**. They use hidden
reservation/value truth and are not fitted, deployable marginal effects.
A 3% proportional reduction in a `0.94` ratio gives `0.9118`, not `0.91`.

## Instruments and evaluator truth

Binary IV uses `recommendation` as Z, actual `cut` as D, `sold_30d` as Y,
and `x` as an observed pre-assignment covariate. Reviewer scores are built
from a separate historical sample. Continuous IV uses `log_offer =
log(offer / baseline)` and outcome `accepted_14d`; its coefficient is probability
units per unit log offer, not percentage points per dollar.

`evaluator_truth` stores private quality, structural values, hidden demand,
reservation values, and simulation draws separately. Explicit feature
allowlists prevent arbitrary oracle columns from entering fitted models.
Tables labeled `oracle_*` deliberately use truth for validation only.
The IV rows' oracle targets refer to their own DGP, not the main resale ATE.

## New diagnostic units and constraints

`assumed_direct_effect` is a probability-unit contribution of changing a binary
recommendation from 0 to 1, holding its mediated treatment pathway separate.
It is not a log-odds coefficient, per-dollar effect, or learned parameter.
Sensitivity rows condition on each assumed value and recompute robust IV/AR
inference; weak instruments can make the adjusted effect highly unstable.

`dr_difference_vs_hold` and `difference_lower/upper` are dollars of lifecycle
contribution per holdout home, using paired score differences. These intervals
are pointwise, not corrected for choosing among multiple candidate policies.
Closing and follow-up dates must be present, and follow-up must extend through
closing before a realized economic reward can be evaluated.

`target_completion` and `predicted_completion` are portfolio-average
probabilities. `shadow_price_per_expected_completion` is the local marginal
contribution cost of increasing expected completions, not the cost per
percentage point of probability. At the maximum supported endpoint this
upward marginal value is not applicable and is left empty.

`randomized_home_fraction` is the share of homes with a nondegenerate lottery
over supported prices. A lottery is not an interpolated quote.
The solver sees fitted probabilities/rewards only; `oracle_*` columns evaluate
its chosen weights afterward. Model-feasible completion is not guaranteed
actual completion. Frontier plots use the fixed holdout portfolio and reference
economics, independently of the app's individual-home stress sliders.

## Provenance and uncertainty

`metadata.json` records configuration, chosen model, temporal boundary,
metrics, resampling counts, and limitations. `manifest.json` records observed
CSV hashes, package versions, and modeling source hashes. Raw datasets and
truth stay ignored by Git; selected synthetic figures/tables are publishable.

Bootstrap intervals refit the randomized response on independent-home resamples,
holding AVM, property profile, and financial assumptions fixed. They are
pointwise, not simultaneous or joint risk guarantees. Stress policies use
five named economic scenarios, not estimated probabilities of future markets.

## Separate linked two-sided extension

The `two-sided` command defaults to `outputs\two_sided`. Its observed
`development.csv`, `continuation.csv`, and `evaluation.csv` each contain one
row per original quoted prospect, including nonacceptors and nonacquirers.
IDs such as `evaluation:0` are globally distinct across these three cohorts.
Join only to the corresponding extension truth by this ID, never by position
or to the original experiment's unrelated numeric IDs.

| Fields or artifact | Contract |
|---|---|
| `offer_date`, `initial_offer`, `initial_outlay`, `initial_offer_ratio` | Original quote and ledger; fixed upstream policy |
| `offer`, `offer_ratio`, `net_proceeds` | Retained initial-quote ledger fields; use the explicit `final_*` fields for the delivered escalation |
| `initial_accepted_7d`, `engagement_score`, `buy_eligible` | Initial response/engagement; escalation eligibility is initial nonacceptance plus engagement |
| `buy_decision_date`, `buy_features_at` | Decision on offer day 7; permitted inputs available before it |
| `repair_forecast`, `margin_forecast`, `max_escalation` | Predecision estimates/cap; no realized repair or resale information |
| `buy_action`, `buy_assignment_probability` | Delivered increase magnitude and known uniform-within-feasible-set probability; full compliance |
| `final_offer`, `final_net_proceeds`, `company_outlay` | Ledger after the logged action; outlay is incurred only if acquired |
| `accepted_14d`, `completed_acquisition_35d` | Escalation-stage endpoints; zero outside that stage is not a missing initial-acceptance label |
| `acceptance_date`, `acquisition_close_date`, `acquired` | Initial or escalation acceptance date, separate close date, and actual acquisition indicator |
| `listing_date`, `early_sold`, `buyer_signal`, `sell_eligible` | Linked acquired-inventory state; markdown requires acquisition, no early close, and signal below 0.65 |
| `sell_decision_date`, `sell_features_at` | Listing day 21 checkpoint and strictly earlier feature availability |
| `base_list_price`, `minimum_list_price` | 1.02 and 0.96 times AVM respectively |
| `sell_action`, `sell_assignment_probability` | Markdown magnitude and known conditional randomization probability |
| `sold_30d`, `sell_mature_at` | Completed sale in the checkpoint's following 30 days; use eligible rows, not already-sold/nonacquired rows |
| `buy_mature_at`, `reward_mature_at`, `observed_through` | Acquisition endpoint and lifecycle observation contracts; all rewards mature by offer day 150 |
| `repair_cost`, `daily_carrying_cost`, `resale_close_date` | Realized acquired-home costs and disposition date; future labels, not acquisition predictors |
| `realized_remaining_net` | After-selling/remaining-carry proceeds for markdown-eligible homes |
| `realized_disposition_net` | After-selling/all-carry proceeds, before acquisition basis and actual repairs |
| `lifecycle_contribution` | Acquired-home net disposition minus basis and repairs, less $200 for each escalation-eligible prospect |
| `continuation_learning.csv` | Cohort-B all-eligible reward labels under downstream hold and optimized resale rules |
| `buy_candidate_values.csv` | Cohort-C candidate quotes, support, completion predictions, and joint values; unsupported predictions are empty |
| `buy_value_curve.csv` | Average candidate values on the same all-four-actions-feasible buy population |
| `buy_decisions.csv`, `buy_action_mix.csv` | Chosen action/gain, forecast cap, and action counts; chosen gain is not maximum candidate gain |
| `stage_effects.csv` | Stage-specific common-support AIPW completion contrasts and matching evaluation-only oracle effects |
| `policy_evaluation.csv` | Sequential DR contribution per escalation-eligible prospect and paired differences vs hold both |
| `cohort_policy_evaluation.csv` | Contribution per original prospect, including fixed noneligible paths; a different denominator |
| `cohort_funnel.csv` | Counts and temporal boundaries for all three cohorts |

The extension's actions are **magnitudes**: `buy_action=0.03` multiplies gross
offer by `1.03`; `sell_action=0.03` multiplies current list price by `0.97`.
Noneligible rows have missing randomization probabilities because no action
was randomized there. Do not fill these with a fictitious propensity or
drop genuine nonacquirers from acquisition reward learning.

Actual repairs are forecast repairs times mean-one lognormal noise. Daily
carry is `20 + 0.00002 * avm_value + 0.08 * company_outlay / 365`.
Closing proceeds average 99% of current list price; terminal proceeds average
94% of hidden value in the generator, with AVM substituted by planners.
Selling expense is 2%. Already-incurred carrying is subtracted when converting
remaining-value scores to acquisition-time value; basis and repairs are then
charged once. Initial marketing/quote expenses are not included.

Acquisition candidate values are learned joint dollar rewards, not acceptance
times unconditional margin. The $150 hurdle concerns the chosen predicted
gain, not a lower confidence bound. Policy intervals are pointwise and
conditional on models fitted entirely before cohort C; no extension bootstrap
or joint uncertainty propagation is claimed.

See the [extension chapter](two_sided_extension.md) for the exact equations,
cohort dates, support restrictions, and limitations, and
[its results](two_sided_results.md) for the distinct seed-42 tables.

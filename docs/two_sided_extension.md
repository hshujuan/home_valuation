# Linked two-sided decisions: acquisition escalation and resale markdown

## Scope

This separate synthetic experiment follows the same prospect through initial
offer, acquisition escalation, listing, resale markdown, and disposition.
It complements rather than replaces the baseline hold-versus-3%-cut resale
experiment described in the [modeling analysis](analysis_report.md).

The decision is whether to increase an engaged seller's initially unaccepted
offer, accounting for the downstream resale policy. That policy must change
the acquisition reward being learned; placing unrelated acquisition and
resale response curves side by side is not sufficient.

The implementation supports one escalation and one markdown under fixed
initial-offer, initial-listing, and terminal-sale mechanisms. It is not a
production negotiation policy or a general dynamic inventory solver. Public
context and methodological references have the limits recorded in
the [source ledger](sources.md), especially S13-S15.

## 1. Timeline and reported populations

One stable `home_id` connects each original quoted prospect to later events.

| Event | Implemented timing and rule |
|---|---|
| Initial quote | Offer day 0; gross offer is 86%, 90%, or 94% of AVM |
| Initial acceptance | Offer days 1-7; acceptors are not escalation-eligible |
| Acquisition decision | Day 7, among initially unaccepted and engaged prospects |
| Escalation | Increase the existing gross offer by 0%, 1%, 2%, or 3%, within its forecast cap |
| Stage acceptance | Within 14 days after the acquisition decision |
| Completed acquisition | Within 35 days after that decision; initial acceptors instead close on offer days 14-25 |
| Listing | Fourteen days after completed acquisition |
| Early resale | May close on listing days 7-19 |
| Markdown decision | Listing day 21, among acquired homes with no early sale and buyer signal below 0.65 |
| Markdown | Cut current list price by 0%, 1%, 2%, or 3%, within its floor |
| Stage resale endpoint | Completed resale in the following 30 days |
| Terminal disposition | Sale 45 days after the markdown checkpoint if not completed within the window |
| Lifecycle maturity | Observed through offer day 150 |

Acceptance is not acquisition. The generator stipulates a 96%
action-independent closing probability conditional on acceptance. The buy
response model learns acceptance and estimates that closing probability from
historical data. The economic reward model does not multiply this response
by an unconditional average margin.

Three different targets are reported:

| Target | Denominator |
|---|---|
| Stage completion contrast versus hold | Evaluation-cohort prospects eligible for all four actions at the specified stage |
| Joint policy contribution | All escalation-eligible prospects, including hold-only prospects and nonacquirers |
| Original-cohort policy contribution | All original quoted prospects, including initial acceptors and nonengaged sellers |

Buy and sell stage contrasts concern different populations. The resale
contrast conditions on acquisition under the logging policy; it is not
automatically the resale effect in inventory selected by a different
acquisition policy. Initial marketing/quote expenses are excluded from the
original-cohort objective.

## 2. Chronology and information boundaries

The default seed-42 run fits the AVM using 5,000 historical market sales and
then generates three cohorts:

| Cohort | Prospects | Role | First offer | Last lifecycle maturity |
|---|---:|---|---|---|
| A: development | 12,000 | Fit stage responses and define the resale rule | 2022-01-30 | 2023-02-03 |
| B: continuation learning | 10,000 | Construct downstream-policy rewards and fit acquisition values | 2023-03-06 | 2024-03-09 |
| C: evaluation | 8,000 | Apply frozen rules and estimate effects and values | 2024-04-09 | 2025-04-13 |

All A lifecycle rewards mature before B starts; all B rewards mature before C
starts. The runner checks this ordering. AVM predictions use strictly prior
historical comparable sales; fitted valuation parameters remain frozen across
the later cohorts.

Buy features include physical attributes, market, pre-offer belief, initial
offer ratio, observed attachment/convenience/engagement scores, forecast repair
and margin ratios, and the predecision escalation cap. Sell features include
physical attributes, market, pre-markdown buyer signal, and list/AVM ratio.
Action terms include magnitude, its square, and limited interactions.

These are explicit allowlists. Future buyer signals, actual repairs, realized
proceeds, acceptance, closing, and hidden value are not acquisition
predictors. Mature historical outcomes can be training labels without becoming
valid decision-time features. The subjective scores are stipulated observable
measurements, not perfect observations of reservation utility.

## 3. Conditional randomization and support

Both stages use positive magnitudes:

$$
\mathcal A=\{0,0.01,0.02,0.03\},\qquad
O(a)=O_0(1+a),\qquad P(b)=P_0(1-b).
$$

Thus the same positive action magnitude increases an acquisition offer but
decreases a resale list price. This differs from the baseline experiment's
signed resale actions.

Each eligible row is randomized uniformly over its predecision feasible set:

$$
e_t(a\mid H_t)=\frac{1}{|\mathcal A_t(H_t)|},
\qquad a\in\mathcal A_t(H_t).
$$

Actions are fully delivered, and scoring verifies the logged probability
against the feasible set. Noneligible rows have missing assignment
probabilities because no stage action was randomized.

The acquisition cap is based on forecast exit proceeds after selling costs,
forecast repairs, initial outlay, and 70 days of forecast carrying. An increase
must leave at least $5,000 of forecast margin after its incremental outlay.
Hold remains feasible because the existing offer cannot be withdrawn by
this policy; that is not evidence it is profitable.

The initial list price is \(1.02\widehat V\), with floor \(0.96\widehat V\).
All four markdowns satisfy the default floor, but row-specific support
checks are still enforced. These are forecast constraints, not realized
profit or volume guarantees.

No instrument is needed when the feasible treatment itself is randomized
and fully delivered. Real overrides would require delivered-action logs,
intention-to-treat reporting, and an additional defensible identification
argument before introducing IV.

## 4. Financial accounting

The shared illustrative offer ledger is:

$$
K(O)=0.98O-6{,}000,\qquad N(O)=K(O)-3{,}000.
$$

\(K\) is company outlay and \(N\) is seller net proceeds. Retained charges and
repair credits already reduce outlay; actual repair spending is separate.
For a $500,000 AVM and 90% initial offer, gross offer is $450,000, outlay
$435,000, and seller net $432,000. A 2% escalation raises outlay by $8,820,
not the full $9,000 gross increase.

Actual repairs equal forecast repairs times mean-one lognormal noise.
Selling expense is 2% of proceeds. Daily carrying combines cash holding
costs and financing:

$$
c_{\mathrm{day}}=
20+0.00002\widehat V+\frac{0.08K}{365}.
$$

Every acquired home pays its basis and actual repairs once. Each
escalation-eligible prospect incurs $200 of fixed follow-up/opportunity cost,
including hold decisions and nonacquisitions. A mature nonacquisition has
genuine reward -$200; a missing disposition for an acquired home is an error,
not a zero-profit observation.

The following is stipulated arithmetic, not a fitted result:

| Quantity | Hold | Raise offer 2% |
|---|---:|---:|
| Completed-acquisition probability | 12% | 15% |
| Selected net exit value after post-acquisition costs | $455,000 | $458,820 |
| Company outlay | $435,000 | $443,820 |
| Conditional lifecycle margin | $20,000 | $15,000 |
| Fixed prospect cost | $200 | $200 |
| Expected contribution per eligible prospect | $2,200 | $2,050 |

Conversion and selected exit value improve, yet expected contribution falls.
The objective is contribution dollars, not gross-margin percentage or
audited enterprise net income.

## 5. Resale continuation value

For history \(H_2\) and markdown \(b\), let \(q_b\) be completed-resale
probability over the next 30 days. `resale_values` computes:

$$
\begin{aligned}
\widehat M_2(H_2,b)
&=0.98\left[q_b\,0.99P_0(1-b)
+(1-q_b)\,0.94\widehat V\right]\\
&\quad-c_{\mathrm{day}}\left[18q_b+45(1-q_b)\right].
\end{aligned}
$$

This is remaining net disposition value. Acquisition basis, actual repairs,
and carrying already incurred before the checkpoint are accounted for
separately. The planner substitutes AVM for the hidden true value used by
the generator. Eighteen days is the conditional mean early-close time;
45 days is the terminal-sale time measured from the checkpoint.

The optimized resale rule maximizes this fitted value over feasible
markdowns. Its response model and economic assumptions are frozen before
cohorts B/C; the word "optimized" describes the rule, not established
performance.

## 6. Learn acquisition value under the downstream policy

For completed acquisition \(C(a)\), outlay \(K(a)\), actual repairs \(R(a)\),
and net disposition \(G^{\pi_2}(a)\) after selling and all carrying under
resale policy \(\pi_2\):

$$
J(a\mid H_1)=
E\!\left[
C(a)\{G^{\pi_2}(a)-K(a)-R(a)\}
\mid H_1
\right]-c_{\mathrm{quote}}.
$$

The joint expectation retains offer-dependent selection into acquired
inventory. It cannot generally be replaced by completion probability times
an unconditional average margin.

For a markdown-eligible home with observed remaining net proceeds \(U_2\),
`stage_scores` supplies the downstream DR score:

$$
\begin{aligned}
\psi_2^{\pi_2}
&=\widehat M_2(H_2,\pi_2(H_2))\\
&\quad+\frac{\mathbf1\{A_2=\pi_2(H_2)\}}{e_2(A_2\mid H_2)}
\left[U_2-\widehat M_2(H_2,A_2)\right].
\end{aligned}
$$

Subtract carrying already incurred between acquisition and markdown to
convert this into an acquisition-time downstream score. For acquired homes
not eligible for markdown, use observed net disposition: the target policy
does not change their resale action.

The resulting pseudo-reward for an escalation-eligible prospect is:

$$
\widetilde R^{\pi_2}
=C\left[\widetilde G^{\pi_2}-K-R\right]-c_{\mathrm{quote}}.
$$

Fit this reward using **all escalation-eligible cohort-B prospects**,
including nonacquirers, as a function of their predecision history and logged
escalation. `fit_continuation` uses standardized features and Ridge regression
with `alpha=50`. Labels are divided by AVM during fitting and predictions
multiplied back into dollars; the objective remains dollar contribution.

Separate models are fitted for downstream hold and optimized resale.
Changing the downstream rule therefore changes acquisition labels and fitted
candidate values. Oracle response probabilities do not enter these models.

## 7. Action selection and sequential evaluation

The acquisition rule has a $150 predicted-gain hurdle and $150 near-optimal
tolerance. It selects the smallest feasible positive increase that is within
$150 of the best fitted value and whose **own gain versus hold** is at least
$150. Otherwise it holds.

For values `[100, 260, 330, 400]`, the 1% action qualifies: its own gain is
$160 and it is $140 below the maximum. Reporting $300 would incorrectly
attribute another action's gain to the chosen action. The hurdle is not a
statistical lower confidence bound or a realized-profit guarantee.

On held-out cohort C, the frozen joint policy is scored as:

$$
\begin{aligned}
\psi_1^{\pi_1,\pi_2}
&=\widehat M_1(H_1,\pi_1(H_1);\pi_2)\\
&\quad+\frac{\mathbf1\{A_1=\pi_1(H_1)\}}{e_1(A_1\mid H_1)}
\left[\widetilde R^{\pi_2}
-\widehat M_1(H_1,A_1;\pi_2)\right].
\end{aligned}
$$

The second-stage residual correction is inside the first-stage corrected
lifecycle reward. Known sequential randomization, conditional support,
consistency, mature rewards, and no interference support evaluation of the
frozen policy even when reward regressions are imperfect. Regression quality
still affects variance and the learned action choices. This follows the
sequential DR principle in S15, not a claim that one-stage conversion fitting
solves dynamic pricing.

Compare policies using paired same-prospect score differences and the sample
standard error of those differences. Intervals are pointwise and conditional
on the learned models; they do not correct unrestricted policy search or
propagate future market uncertainty.

For original-cohort values, replace escalation-eligible prospects' rewards
with their first-stage scores and retain downstream-corrected rewards on
noneligible paths. Initial acceptors can still be affected by the resale
policy, but are not charged the escalation opportunity cost. This explains
why the original-cohort difference need not equal the eligible-only
difference times the eligible share.

The reported path-weight effective sample size uses the product of relevant
matching-action weights. It diagnoses weight concentration; it does not
replace confidence intervals.

## 8. Reference results and limitations

The [generated results](two_sided_results.md) contain the full precision
tables. In the 8,000-prospect evaluation cohort, 4,214 prospects are
escalation-eligible, 1,060 homes are acquired under logging, and 750 qualify
for markdown.

| Finding | Interpretation |
|---|---|
| Acquisition +3% completion lift: +3.52 percentage points, interval [0.88, 6.17]; oracle +3.87 | Uses the 2,457 buy prospects supporting all four actions and the completed-acquisition endpoint |
| Resale 3% cut lift: +21.06 points, interval [11.66, 30.46]; oracle +8.96 | Uses 750 markdown-eligible homes; this realization's interval misses its target |
| Selective acquisition holds all 4,214 eligible offers | `linked_selective` and `resale_only` coincide despite genuinely different downstream-conditioned reward learning |
| Linked-selective contribution change: -$273.68 per eligible prospect, interval [-$1,287.60, $740.24] | Does not establish improvement |
| Linked-selective change: +$4.65 per original prospect, interval [-$713.21, $722.52] | Includes changed initial-acceptor resale paths and a different denominator |

Hold-both has modeled contribution about -$913.15 per original prospect.
The fixed upstream offer system is not demonstrated to be profitable.
No seed or generator parameter was selected to force positive escalation
or a policy improvement.

![Linked stages and paired policy comparisons](figures/two_sided_extension.png)

The experiment does not optimize initial quotes, repeated negotiation,
marketing, renovation decisions, capital allocation, spillovers, or a dynamic
terminal policy. Closing conditional on acceptance, repair uncertainty,
terminal values, and timing mechanisms are stipulated. Real systems also
need cancellation/censoring models, jointly calibrated uncertainty, and
assignment/dependence-aware inference. Action support is not a guarantee of
fairness, compliance, profitability, or external validity.

## 9. Implementation and execution

| Component | Responsibility |
|---|---|
| `two_sided_simulation.py` | Stable IDs, lifecycle chronology, feasible actions, randomization, observed outcomes, and separate truth |
| `stage_features`, `fit_stage_response` | As-of feature contracts and completion response models |
| `resale_values`, `supported_best` | Remaining-value economics and supported resale selection |
| `continuation_rewards`, `fit_continuation` | Downstream-corrected labels and acquisition joint-value learning |
| `select_escalation` | Actual chosen-gain hurdle and smallest near-optimal increase |
| `stage_scores`, `stage_effects` | Known-propensity scores and stage-specific common-support contrasts |
| `run_two_sided`, `save_two_sided` | Cohorts, fixed-policy evaluation, persisted outputs, and generated reports |
| `tests/test_two_sided.py` | Timing, support, score algebra, accounting, chosen gains, and downstream linkage |

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m home_valuation two-sided --seed 42 --write-report
.\.venv\Scripts\python.exe examples\09_linked_two_sided.py
```

Default outputs are written to `outputs\two_sided`, not `outputs\reference`.
`--write-report` updates only the linked generated Markdown and figure; it
does not rewrite this chapter. For a compact run, use
`--size 2400 --output outputs\two_sided_smoke`. Bootstrap/Monte Carlo flags
belong to the baseline command and are rejected here.

The app uses `max(3600, 2 * selected size)` prospects per linked cohort and the
selected size for market history. Its results need not equal the default
CLI reference. See the [data dictionary](data_dictionary.md) for field
contracts, the [architecture](architecture.md) for software boundaries, and
the [README](../README.md) for installation and the separate summary-PDF build.

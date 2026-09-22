# Home valuation and two-sided causal decisions

## 1. Decision architecture

The system is best understood as **two-sided decision-making built on a
valuation distribution**, not a single price predictor. Valuation and
underwriting describe the asset and its uncertainty. Causal response models
describe how feasible actions change homeowner and buyer behavior. The
decision layer combines these responses with lifecycle economics, risk,
and available capital.

**Valuation is not the purchase offer, and neither is the resale list price.**

```text
                    VALUATION / UNDERWRITING
       Value distribution | condition | repairs | uncertainty
                              |
              +---------------+---------------+
              |                               |
       BUY / ACQUISITION                 SELL / RESALE
         Initial offer                    Initial list
              |                               |
         Seller signal                    Buyer signal
              |                               |
       Upward escalation                Downward markdown
              |                               |
              +---------------+---------------+
                              |
               PORTFOLIO ECONOMICS / RISK / CAPITAL
```

This summary presents the framework, its causal interpretation, and an
implemented synthetic prototype. Public information motivates the context;
numerical findings are synthetic, not estimates of Opendoor's actual response
curves, internal policy, or operating performance. The complete implementation
handbook remains available separately as `design_implementation_report.pdf`.

| Quantity | Meaning | Decision implication |
|---|---|---|
| Valuation distribution | Possible market/exit values under a stated date, condition, and transaction definition | Foundation for underwriting; not itself a customer quote |
| Purchase offer | Gross acquisition quote, with seller net proceeds determined by explicit terms | Trades off acquisition probability, selected inventory, cost, and risk |
| Resale list price | An asking-price action, distinct from eventual transaction proceeds | Trades off buyer response, negotiation, time to close, and retained inventory value |

For example, a **stipulated** value distribution centered near $500,000 can
coexist with a $450,000 acquisition offer and a $510,000 initial list price.
These are not inconsistent valuations: the quotes serve different sides of
the business and incorporate costs, risk, competition, and response.
They are illustrative numbers, not recommended spreads.

## 2. Valuation and underwriting as a distribution

Define the target before selecting an algorithm:

$$
F_V(v\mid X,t,s)=P(V_{\mathrm{exit}}\le v\mid X,t,s).
$$

\(X\) contains point-in-time property and market information, \(t\) is the
valuation date, and \(s\) specifies condition, sale horizon, and market scenario.
As-is market value, renovated exit value, and the price from a forced
disposition are different targets. A repair cost is not automatically an
equal increase in market value.

The model should supply a central estimate and useful uncertainty information,
such as calibrated quantiles or scenario distributions. Underwriting also
requires forecasts of repairs, holding time, selling expense, and financing.
Their errors can be correlated: a market decline can simultaneously reduce
exit value, slow completion, and increase capital usage.

**Model development.** Start with comparable sales and a hedonic/log-price
baseline, then evaluate nonlinear models against the same future holdout.
Construct comparable-sale features only from transactions available at the
decision. Preserve the distinction between homeowner-reported and verified
condition, including when each became available. A later inspection cannot
retroactively justify an earlier quote.

**Validation.** Report dollar and relative error, directional bias, tail loss,
interval coverage, and performance by geography, price range, condition, and
market period. Evaluate acquired inventory separately: good accuracy across
all prospects does not imply good accuracy on the selected purchased pool.
Nominal interval coverage is not a guarantee under temporal or geographic
shift.

The prototype illustrates this with a future-period hedonic-model MAE of
**$25,384**, versus **$32,384** for its boosted alternative. Nominal 90%
intervals cover **86.21%** of future observations. Complexity does not
automatically improve forecasting, and uncertainty calibration needs monitoring.
These numbers describe the baseline synthetic experiment only.

**Implementation boundary.** The current code fits point predictions and
split-conformal intervals; it does not estimate a fully calibrated joint
distribution of exit value, repairs, time, and market shocks. That richer
distribution is an architectural objective, not an implemented capability.

Public offer descriptions identify comparable sales, market conditions,
verified condition, and human review as relevant inputs [1]. They do not
establish a particular model architecture or forecasting accuracy.

## 3. Acquisition: initial offer, seller signal, escalation

The company's buy decision acts on the homeowner's willingness to sell.
Three quantities should remain separate: the owner's belief about market
value, the property's market value, and the owner's reservation utility.
The reservation utility also reflects outside options, transaction costs,
convenience, certainty, and attachment. A refusal does not by itself prove
the valuation model is wrong; an acceptance does not validate the quote as
market value.

A useful behavioral representation is

$$
A(O)=\mathbf 1\{N(O)+B_{\mathrm{convenience}}
\ge R+\varepsilon\},
$$

where \(O\) is the gross offer, \(N(O)\) the seller's net proceeds, and \(R\)
reservation utility expressed in monetary units. Completed acquisition is
a separate event after acceptance. The distinction matters whenever deals
cancel or closing conditions change.

**Initial offer.** Use the valuation distribution, expected costs, and
capital/risk limits to choose a feasible quote. Learning its causal effect
requires logging all offered opportunities, not only accepted transactions.
The target may be completed acquisitions or expected contribution per
original prospect, with a specified closing horizon.

**Seller signal.** Initial acceptance, continued engagement, a counteroffer,
verified condition, and stated expectations can provide additional information.
However, these signals are partly responses to the initial quote. Conditioning
on them when estimating the initial offer's total effect can remove an
important causal pathway or introduce selection bias.

**Upward escalation.** Once the seller signal has been observed, it becomes
part of the history for the next decision. A stage-specific question is the
effect of an incremental offer increase among initially unaccepted, still-engaged
prospects. Its estimand is narrower than the effect of increasing every initial
offer. If the upstream policy changes, the escalation population can change too.

The acquisition objective is a joint expectation:

$$
J_{\mathrm{buy}}(a\mid H)=
E\!\left[
C(a)\{G^{\pi_{\mathrm{sell}}}(a)-K(a)-R_{\mathrm{cost}}(a)\}
\mid H
\right]-c_{\mathrm{followup}}.
$$

\(H\) is observed pre-escalation history; \(C\) is completed acquisition;
\(K\) company outlay; \(R_{\mathrm{cost}}\) actual repairs; and
\(G^{\pi_{\mathrm{sell}}}\) downstream net disposition after selling and carrying
costs under the resale policy.

Increasing the offer changes both the acquisition rate and **which homes are
acquired**. Consequently, completion probability multiplied by an unconditional
average acquired-home margin is generally insufficient. Historical mature
rewards on all eligible prospects can instead support learning the joint
objective directly.

The housing-intermediation literature motivates this valuation/selection
trade-off [2]. Work on homeowner expectations also helps explain why beliefs
affect selling behavior without being interchangeable with a property's
physical value [3]. Neither supplies this prototype's response coefficients.

## 4. Resale: initial list, buyer signal, markdown

The sell decision acts on buyers, so its usual response direction differs
from acquisition: a lower list price can increase buyer interest and
completion, but can also reduce transaction proceeds.

**Initial list.** Combine the exit-value distribution with expected response,
time to close, negotiation, and holding costs. List price is an intervention,
not the target label for a market-value model. An initial-listing experiment
should define its population and follow-up from listing, rather than treating
it as the same experiment as repricing already-unsold inventory.

**Buyer signal.** Views, tours, offers, elapsed time, and comparable-market
activity update the decision state. Weak engagement may reflect an excessive
price, weak demand, property condition, marketing, or a combination. Historical
markdowns are therefore often concentrated on harder-to-sell inventory.
A negative historical association between cuts and completion is not evidence
that cuts causally reduce completion.

**Downward markdown.** At a defined checkpoint, estimate effects among homes
still eligible and unsold. Use signals observed before that checkpoint.
Post-markdown views, offers, or negotiations are mediators/outcomes, not
pretreatment controls for the markdown's total effect.

Let \(q_b(H)\) be the probability of completed resale by horizon \(h\) under
markdown \(b\). A continuation-aware objective is

$$
\begin{aligned}
J_{\mathrm{sell}}(b\mid H)
&=q_b(H)\,E[U_{\mathrm{early}}\mid \mathrm{close},b,H]\\
&\quad+(1-q_b(H))\,E[U_{\mathrm{continue}}\mid
\mathrm{no\ close},b,H].
\end{aligned}
$$

The two \(U\) terms are net future disposition values, including relevant
selling and carrying costs. Acquisition basis is already sunk for incremental
repricing comparisons, but must still be charged once when reporting lifecycle
profit. A home unsold at the measurement horizon retains an asset value.

The prototype uses a fixed day-45 terminal sale for clarity. A production
system would need a defensible continuation policy and richer timing/cancellation
models. Repeated markdowns create a sequential decision problem: effects
conditional on historical prices cannot simply be reused as if future
inventory composition and buyer signals were unchanged.

## 5. Causal design across both sides

Valuation is primarily a forecasting task. Causal inference identifies how
actions built on that forecast change response, selection, proceeds, and time.
It also informs underwriting by revealing the economics of selected inventory;
it does not turn observed accepted offers into unbiased market-value labels.

| Design element | Acquisition | Resale |
|---|---|---|
| Decision population | Original offered prospects, or a separately defined engaged-nonacceptor population | Newly listed inventory, or a separately defined unsold-at-checkpoint population |
| Action | Initial quote or incremental upward escalation | Initial list or incremental downward markdown |
| Response | Acceptance and completed acquisition, separately | Interest, offer/contract, and completed resale, separately |
| Economic outcome | Contribution per eligible prospect, including nonacquisitions | Remaining value and lifecycle contribution, including unsold inventory |
| Main selection issue | Who requests, remains engaged, accepts, and closes | Who was acquired and remains unsold at the checkpoint |

Prefer experiments over operationally feasible actions, with logged assignment
probabilities and delivered terms. Randomize within predecision constraints,
such as offer caps and list-price floors. A contrast is identified only where
both actions have support; a constrained cohort cannot support a universal
recommendation outside its feasible range.

Use outcome regression for precision and doubly robust estimation to combine
predictions with propensity-weighted residual corrections. Cross-fitting
separates nuisance estimation from scoring. Under observational assignment,
this still requires adequate overlap and no unmeasured confounding; flexible
machine learning does not supply those assumptions.

**Instrumental variables:** where credible assignment affects a delivered
price but not outcomes through other pathways, instrument-based methods may
identify a narrower causal response. They require additional design assumptions.
Detailed instrument selection and diagnostics are outside this summary.
The linked prototype randomizes fully delivered actions and does not need
this additional machinery.

Keep policy development separate from final evaluation. Estimate differences
between fixed policies on the same held-out prospects, with uncertainty on
the paired differences. Do not infer superiority by comparing two independent
intervals or by selecting the most favorable result from many policies.
Missing or immature outcomes are not negative labels.

## 6. Connect the stages through downstream value

The implemented extension follows stable prospect IDs through acquisition,
listing, markdown, and disposition. It has one escalation and one markdown;
the initial-offer and initial-list mechanisms are fixed upstream inputs.

| Stage | Implemented contract |
|---|---|
| Acquisition escalation | Day 7 after the initial quote, for engaged nonacceptors; feasible increases of 0/1/2/3% |
| Acquisition outcomes | Acceptance within 14 days and completed acquisition within 35 days of escalation |
| Resale markdown | Listing day 21, conditional on acquisition, no early sale, and weak buyer signal; feasible cuts of 0/1/2/3% |
| Resale outcomes | Completed resale over the next 30 days; otherwise terminal disposition at day 45 |
| Complete reward | Observed through offer day 150; company outlay, repairs, selling, and carrying accounted for once |

Three chronological cohorts separate learning and evaluation:

1. **Development: 12,000 prospects.** Fit stage responses and define a resale
   rule from observable information.
2. **Continuation learning: 10,000 prospects.** Use mature downstream outcomes
   and known resale propensities to construct policy-adjusted rewards; fit
   acquisition joint values on all escalation-eligible prospects.
3. **Evaluation: 8,000 prospects.** Freeze the learned rules, apply sequential
   doubly robust scores, and compare lifecycle contribution.

Every earlier cohort's lifecycle outcomes mature before the next cohort
begins. The shared AVM is fitted on separate earlier market transactions.
At each eligible resale decision, the downstream score combines predicted
value under the target action with an inverse-propensity correction from
the logged action. The acquisition score applies the same correction to
the resulting lifecycle reward. This connects downstream policy to acquisition
learning without using future outcomes as decision inputs [4,5].

The implementation uses allowlisted features, as-of forecast margin caps,
known conditional action support, and separate evaluator-only truth.
The acquisition rule selects the smallest near-optimal feasible increase
whose **own predicted gain** meets a $150 hurdle. A maximum-candidate gain
cannot be reported as the gain of a different chosen action. Holding every
offer is a valid outcome.

## 7. Portfolio economics, risk, and capital

Individual conversion and margin curves are inputs to a portfolio decision,
not its final objective. A coherent formulation is

$$
\begin{aligned}
\max_{\pi_{\mathrm{buy}},\pi_{\mathrm{sell}}}\quad&
E\left[\sum_i \Pi_i\right]
-\lambda\,\mathcal R\left(\sum_i\Pi_i\right)\\
\text{subject to}\quad&
\text{capital, concentration, service, and action constraints}.
\end{aligned}
$$

\(\Pi_i\) is lifecycle contribution and \(\mathcal R\) a specified downside-risk
measure. This is a **proposed general formulation**, not a claim that the
current prototype implements a full capital/risk optimizer.

| Layer | Required discipline |
|---|---|
| Accounting | Separate seller net proceeds, company outlay, actual repairs, selling, holding, and financing; charge each component once |
| Capital | Model cash deployed over time, not only average completed acquisitions; cancellation and holding-time uncertainty affect exposure |
| Portfolio risk | Preserve shared market shocks and concentration; independent-home uncertainty understates correlated downside |
| Objective | Distinguish contribution dollars, gross profit, and margin percentage; maximizing the ratio can sacrifice scale |
| Constraints | Use supported actions and explicit business limits; fitted expected-volume feasibility is not a hard realized guarantee |

In the **illustrative ledger**, a $450,000 gross offer with a 2% retained charge
and $6,000 repair credit gives $435,000 company outlay. A separate $3,000 seller
closing expense leaves $432,000 seller net. Actual repair spending remains
another cost. These are synthetic assumptions, not a statement of current
company fees; public documentation describes variable charges and distinct
offer components [6].

The existing portfolio example trades expected completion against contribution
using lotteries over tested prices. It does not interpolate an unsupported
average quote. It also does not enforce realized cash-path limits, jointly
optimize both sides' capital, or solve a full dynamic inventory-control problem.

In its synthetic reference frontier, increasing fitted expected completion
from **32.46% to 42.56%** reduces fitted contribution from roughly **$40,485
to $38,975 per home**. This is an economic trade-off under model estimates,
not a forecast guarantee or a capital-risk stress result.

## 8. Illustrative evidence and interpretation

The results below come from the saved seed-42 experiments. They demonstrate
failure modes and decision trade-offs, not real-market policy effectiveness.
The baseline and linked experiment have different populations and assumptions.

| Finding | Interpretation |
|---|---|
| Baseline randomized resale completion lift: +8.95 percentage points; naive observational difference: -5.80 points | Operational selection into cuts can reverse the historical association |
| One baseline home's 3% cut raises fitted completion from 17.86% to 25.93%, while contribution falls from $32,358 to $30,559 | A positive response effect does not imply an economically attractive action |
| Linked acquisition +3%: estimated completion lift +3.52 points, 95% interval [0.88, 6.17], matching oracle +3.87 points | Stage-specific effects concern completed acquisition within the defined eligible population |
| Linked resale 3% cut: estimated lift +21.06 points, interval [11.66, 30.46], matching oracle +8.96 points | This interval misses its target; finite-sample uncertainty and model error remain material |
| Linked selective rule holds all 4,214 escalation-eligible offers; estimated value difference vs hold-both is -$274 per eligible prospect, interval [-$1,288, $740] | The experiment does not establish a joint-policy profit improvement; the rule must be allowed not to escalate |

The matching acquisition contrast uses 2,457 prospects with support for every
action; the resale contrast uses 750 markdown-eligible homes. These are not
the same population or one combined effect. Oracle values are known only
inside the simulator and never supply decision features.

For all 8,000 original linked prospects, including initial acceptors, the
selective-rule difference is approximately +$5 per prospect, with interval
[-$713, $723]. The denominator and affected paths differ from the eligible-only
comparison. Absolute modeled contribution is negative under this fixed
upstream system; downstream optimization does not establish business viability.

The policy intervals are pointwise and conditional on frozen learned rules.
They do not incorporate every form of policy search, shared market uncertainty,
or parameter drift. Unfavorable results are retained rather than selecting
parameters or seeds to manufacture a winning policy.

## 9. Implementation and operating priorities

The prototype shares one Python implementation across command-line execution,
generated reports, examples, and a local interactive application. It keeps
the baseline experiment and the linked extension separate.

| Component | Implemented responsibility |
|---|---|
| `features.py`, `valuation.py` | Point-in-time comparable sales, model selection, chronological evaluation, and intervals |
| `seller.py`, `causal.py` | Offer accounting, homeowner response/selection examples, and causal response estimation |
| `optimization.py`, `evaluation.py` | Continuation-aware economics, supported policies, portfolio completion trade-offs, and evaluation |
| `two_sided_simulation.py`, `two_sided.py` | Linked trajectories, mature reward learning, action constraints, and sequential policy evaluation |
| `pipeline.py`, CLI, examples, app | Reproducible artifacts and shared presentation, without a separate dashboard-only model |

Reference artifacts are in `outputs\reference` and `outputs\two_sided`.
The maintained design, contracts, sources, and generated results remain in
`docs`. The summary does not replace the detailed handbook or reproduce
all diagnostics.

```powershell
python -m home_valuation pipeline --seed 42 --output outputs\reference
python -m home_valuation two-sided --seed 42 --output outputs\two_sided
python -m streamlit run app\streamlit_app.py
```

Before operational use, prioritize four workstreams:

1. **Measurement and logging.** Capture original risk sets, action assignments,
   delivered terms, signals, timestamps, overrides, cancellations, and complete
   cash flows, including nonconverting opportunities.
2. **Causal learning.** Validate feasible stage experiments; evaluate changes
   to initial policies separately; handle treatment-induced signals, shifting
   eligibility, and dependence across properties and markets.
3. **Risk integration.** Calibrate joint value/cost/time scenarios and connect
   both sides to capital exposure, concentration limits, and explicit downside
   objectives.
4. **Monitoring and governance.** Track value residuals, selected-inventory
   error, completion calibration, action support, reward maturity, and realized
   economics. Review customer treatment and legal obligations independently
   of model accuracy.

The organizing principle is stable across implementations: **forecast asset
value, identify how actions change behavior and selection, and allocate
capital using the resulting distribution of lifecycle outcomes**. A valuation,
a purchase offer, and a resale list price remain distinct throughout.

## Selected references

These sources provide context and methodology, not empirical calibration of
the synthetic results. Detailed verification limits are recorded in the
separate source ledger.

1. [Opendoor: How does Opendoor determine my offer price?](https://help.opendoor.com/selling/understanding-your-offer/how-offer-price-determined)
   Public valuation inputs and assessment context.
2. [Buchak et al.: Why is Intermediating Houses so Difficult? Evidence from iBuyers](https://www.nber.org/papers/w28252)
   Housing intermediation, valuation, and selection.
3. [Bottan and Perez-Truglia: Betting on the House](https://www.nber.org/papers/w27412)
   Subjective expectations and selling behavior; not an offer-price elasticity.
4. [Dudik, Langford, and Li: Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601)
   Combining response prediction with propensity correction.
5. [Jiang and Li: Doubly Robust Off-policy Value Evaluation for Reinforcement Learning](https://proceedings.mlr.press/v48/jiang16.html)
   Extension of policy evaluation to sequential decisions.
6. [Opendoor: What's included in my offer?](https://help.opendoor.com/selling/understanding-your-offer/whats-in-your-offer)
   Offer components, charges, and seller net proceeds.

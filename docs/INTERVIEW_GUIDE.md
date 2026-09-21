# Interview guide: "1. the valuation model, 2. how you think about the causal component"

## The 90-second version

> "I'd separate a **prediction** problem from an **intervention** problem.
>
> **Valuation** is prediction: estimate the home's value at resale with calibrated uncertainty — comps and a
> repeat-sales index for the anchor, a hedonic/ML ensemble on log price, quantile or conformal intervals, human review
> where the model is unsure, and an HPA forecast over the hold. The key subtlety is that accuracy on all homes is not
> accuracy on the homes we buy: sellers know their home better than we do, so the offers that get accepted are
> disproportionately the ones where we over-valued. The spread has to price that in, and it should widen with
> uncertainty.
>
> **The causal component** is: if we move price, what happens to conversion? Price isn't randomly assigned — analysts
> give better offers on homes they can see are nicer, and those sellers are also harder to convert. So a regression of
> acceptance on spread finds almost no effect, and an optimizer trusting it would widen spreads until volume collapses.
> I'd identify the elasticity with randomized price offsets first, use quasi-experiments we already own — analyst
> leniency, algorithm rollouts — as corroboration, estimate heterogeneity by seller segment, and plug the causal curve
> into an expected-margin objective: P(accept | spread) × margin(spread), with volume targets as a constraint whose
> shadow price tells us what a marginal acquisition costs."

## The 10-minute version (whiteboard order)

1. **Unit economics** (write it): E[π] = A(offer) × (E[sale | list] − offer − repairs − selling − holding·E[days]).
2. **Valuation**: target log price; layers (comps/index → hedonic/GBM ensemble → intervals); metrics (MdAPE, within-5%,
   coverage, error by uncertainty bucket, out-of-time); human-in-the-loop; HPA over the hold.
3. **Adverse selection**: E[log V/V̂ | accept] < 0. Measure it on realized resales; widen spread with uncertainty.
4. **Draw the DAG** (U → spread, U → reservation; instruments → spread only).
5. **Identification ladder** — experiment → IV (leniency, rollout) → DML for observables → HTE.
6. **Instrument checklist** for each candidate: relevance (first-stage F), independence (balance tests), exclusion
   (what else does it touch?), monotonicity, what population it speaks to (LATE).
7. **Worked example**: 3% change → ΔP ≈ β·P(1−P)·Δ; show both sides (below).
8. **Optimization**: FOC/markup rule, winner's-curse slope, buyer-side DOM/pass-through tradeoff, Lagrangian for volume.
9. **Uncertainty & learning**: robust choice over elasticity posterior, always-on small experiments, guardrails.

## Numbers from the repo you can quote ("in a simulation I built…")

| | naive | causal | truth |
|---|---|---|---|
| +1pp spread → acceptance | −0.27pp | −3.01pp (experiment IV) | −3.04pp |
| DOM elasticity θ | −0.15 | 12.3 | 12.0 |
| Optimal spread change | +6pp (grid edge) | +3.0pp | +3.0pp |
| Optimal list multiplier | 1.08 (grid edge) | 1.015 | 1.025 |
| Regret vs true optimum | 8% (seller), 26% (buyer) | ~0% / ~1% | — |

3% list cut: P(sold ≤ 60d) +12pp, ~16 fewer days on market, but −$3.4k contribution per home at today's price level —
only worth it if inventory risk or a volume/capital constraint binds.

## "Customer self-appreciation" — how to handle it live

> "I took that to mean the seller's own valuation — how much they appreciate their home relative to the market. It's
> central to conversion: acceptance is essentially P(offer ≥ seller's net reservation value), and that reservation is
> the market value inflated by self-appreciation — endowment effect, anchoring on online estimates, and extrapolating the
> neighborhood's recent price growth. It shifts the acceptance curve, it's why adverse selection happens (it's anchored on
> the true value, which the seller knows better than us), and we can measure it by asking sellers what they think the home
> is worth and comparing to our valuation. Did you mean that, or appreciation of the asset during our hold? That second one
> matters for how we price holding time and risk."

## Likely follow-ups

**Why not just A/B test everything?** Price tests cost margin on every off-optimal offer, have limited power per segment,
can contaminate local comps (SUTVA), and go stale when the market moves. Use small, stratified, always-on offsets, and
use quasi-experiments to extend coverage.

**Defend the analyst-leniency instrument.** Show assignment is as-good-as-random (balance of pre-determined features
across analysts), control for other levers analysts touch (repair estimates, follow-up speed), leave-one-out construction,
check monotonicity by comparing first stages across subgroups, and interpret as LATE for reviewed offers.

**What if the over-ID test passes?** It only tells you the instruments agree. They could share a bias. Anchor on the
randomized arm.

**How is DML different from IV?** DML handles many *observed* confounders flexibly; it cannot handle unobserved ones.
IV handles unobserved confounding using an exogenous shifter. They combine (DML-IV / DRIV) for nuisance functions.

**Binary outcome — 2SLS or something nonlinear?** 2SLS on the LPM for a robust average effect; control-function logit
or a structural reservation-value model when you need the whole curve for optimization; validate the curve against the
experiment arms.

**How would the spread change in a downturn?** HPA falls and its variance rises → holding time is costlier → the
buyer-side optimum shifts to lower list prices and the seller-side optimum to wider spreads; the buy-box should tighten
toward liquid, easy-to-value homes where adverse selection is smallest.

**What breaks this?** Stale elasticities, feedback (our prices move comps that feed the OVM), strategic sellers who learn
the offer function, and optimizer extrapolation beyond the range the experiments covered. Guardrails and continuous
experiments are the mitigation.

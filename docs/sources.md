# Public source and claim ledger

This report combines public-source research, standard statistical derivations,
and independently written synthetic experiments. Private source files are not
distributed, and results from unexecuted external example projects are not
presented as this lab's output.
All results described as "our simulation" come from this repository.

Verification statements below describe the original source review, not a new
live verification whenever the code or reports are rebuilt. Public commercial
terms and bibliographic revisions should be checked at their linked sources
before relying on them operationally.

## Company context

| ID | Source | What it supports | What it does not establish |
|---|---|---|---|
| S1 | [Opendoor: How does Opendoor determine my offer price?](https://help.opendoor.com/selling/understanding-your-offer/how-offer-price-determined) | Public description of comparable sales, market conditions, verified home condition, automated valuation, and human review | Internal architecture, coefficients, experiment design, model accuracy, or causal identification |
| S2 | [Opendoor: What's included in my offer?](https://help.opendoor.com/selling/understanding-your-offer/whats-in-your-offer) | Distinct offer components, service charge, condition adjustment, closing costs, and net proceeds; the page says the service charge varies | A universal 5% fee, or the synthetic 2% retained-service assumption used here |
| S3 | [Opendoor: What is the home assessment?](https://help.opendoor.com/selling/getting-your-offer/home-assessment) | A real operational self-assessment process using homeowner photos, and an in-person alternative; condition verification before a final offer | That self-reported condition is independently verified or an unbiased valuation input |

The substantive text of these three official pages was retrieved and read.
Product details can change; consult the live pages rather than treating the
simulator's fee schedule or timelines as current commercial terms.

## Housing economics and homeowner behavior

| ID | Source | Supported interpretation and verification |
|---|---|---|
| S4 | Buchak, Matvos, Piskorski, and Seru, [Why is Intermediating Houses so Difficult? Evidence from iBuyers](https://www.nber.org/papers/w28252), NBER Working Paper 28252; December 2020, PDF revised June 2025 | Read the original PDF abstract and introduction. Its central trade-off connects fast liquidity provision with valuation accuracy and adverse selection; it emphasizes relatively liquid, easier-to-value homes. This motivates selection-aware economics, not a calibrated parameter in our simulator. |
| S5 | Bottan and Perez-Truglia, [Betting on the House: Subjective Expectations and Market Choices](https://www.nber.org/papers/w27412), NBER Working Paper 27412; June 2020, PDF revised April 2024 | Read the original PDF abstract and introduction. The randomized information experiment supports a causal role for homeowner appreciation expectations in selling behavior. It is an information/expectations study, not an experiment changing an iBuyer's cash offer by 3%. |
| S6 | Genesove and Mayer, [Loss Aversion and Seller Behavior: Evidence from the Housing Market](https://www.nber.org/papers/w8143), NBER Working Paper 8143 (2001); journal DOI [10.1162/003355301753265561](https://doi.org/10.1162/003355301753265561) | Verified the NBER title, authors, identifier, and working-paper citation. The older PDF's text extraction was garbled; the report uses this as a qualitative literature reference and does not import numerical estimates. The correct working paper is **8143**, not 8146. |

The report's account of S4 and S5 is an original synthesis of inspected
introductory material, not a claim to have replicated their empirical work.
No paper's estimated coefficient is used as the simulator's price elasticity.

## Identification, inference, and policy evaluation

| ID | Source | Use and verification scope |
|---|---|---|
| S7 | Chernozhukov et al., [Double/Debiased Machine Learning for Treatment and Causal Parameters](https://arxiv.org/abs/1608.00060) | Verified the original paper locator and title. Methodological reference for orthogonal scores and cross-fitting. Our AIPW score and its assumptions are implemented in `src/home_valuation/causal.py` and checked in tests; this is not a replication of the paper's empirical examples. |
| S8 | Dudik, Langford, and Li, [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601) | Verified the original locator and title. Reference for combining outcome predictions with inverse-propensity residual correction. Our application is a held-out, finite-action randomized logging policy, with known probabilities. |
| S9 | Angelopoulos and Bates, [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511) | Verified the original locator and title. Background for split-conformal prediction and the importance of exchangeability; not evidence that nominal coverage survives our time shift. |
| S10 | Goldsmith-Pinkham, Hull, and Kolesar, [Leniency Designs: An Operator's Manual](https://www.nber.org/papers/w34473), NBER Working Paper 34473; November 2025, PDF revised June 2026 | Read the original PDF abstract and opening discussion. Supports careful assignment, exclusion, monotonicity, and generated-instrument analysis. The paper discusses UJIVE; our small historical-score teaching fixture does **not** implement UJIVE. It also cautions against automatic clustering prescriptions: inference must follow the assignment and dependence structure. |
| S11 | [`linearmodels` first-stage diagnostics API](https://bashtage.github.io/linearmodels/iv/iv/linearmodels.iv.results.FirstStageResults.diagnostics.html) | Read the diagnostic definitions. With robust covariance, `f.stat` is a Wald statistic with a chi-squared reference distribution; report `f.dist` rather than silently calling every statistic an ordinary F. |

Some HTML fetches for academic papers exposed only their title/citation, not
full text. These entries distinguish verified bibliographic locators from
inspected full-text sections. No inaccessible empirical result is presented
as verified, and no generated citation tokens are used as references.

## Governance

**S12.** [OCC Bulletin 2024-17: Automated Valuation Models, Final Rule](https://www.occ.gov/news-issuances/bulletins/2024/bulletin-2024-17.html).
Read the official summary and highlights. It addresses AVM quality controls
for specified mortgage-originator/secondary-market uses, including confidence,
data manipulation, conflicts, testing, and nondiscrimination. It does not
mandate a particular causal algorithm, certify this project, or automatically
establish the rule's application to every iBuyer transaction.

## Provenance rules used in the report

1. **Public fact:** a limited claim tied to the source above.
2. **Derived theory:** a stated equation with its assumptions and units.
3. **Stipulated illustration:** deliberately chosen numbers, not a fitted estimate.
4. **Fitted synthetic result:** produced by the documented seed, data sizes,
   code, and estimator; see [reference results](reference_results.md).
5. **Oracle diagnostic:** uses hidden simulator truth only for evaluation.
6. **Proposed extension:** a future production/research step, not implemented or
   empirically proven here. The explicitly named **linked two-sided extension**
   is now implemented as a separate synthetic study, not a production validation.

An information ITT, a complier-specific IV effect, a local continuous slope,
and a 30-day resale ATE remain different estimands even when all are described
informally as "causal effects."

## Linked two-sided extension: additional sources

| ID | Source | Supported claim and verification limit |
|---|---|---|
| S13 | [Opendoor: Why did my offer change?](https://help.opendoor.com/selling/understanding-your-offer/offer-changed) | Official page text inspected during the separate two-sided review. Supports assessment-related offer changes, not a documented policy of raising every engaged seller's offer by 1-3%. |
| S14 | [Opendoor: Negotiating price](https://help.opendoor.com/buying/making-an-offer/negotiating-price) | Official page text inspected during that review. Supports negotiation considerations including market conditions and time on market, not a causal markdown elasticity or the simulator's fixed day-21 rule. |
| S15 | Jiang and Li, [Doubly Robust Off-policy Value Evaluation for Reinforcement Learning](https://proceedings.mlr.press/v48/jiang16.html), ICML 2016, PMLR 48:652-661 | Verified the proceedings citation and read its abstract for the extension. Methodological reference for extending DR evaluation to sequential decisions; the explicit two-stage equations here are independently derived and checked, not an empirical replication or a claim to have audited the full paper. |

These additional sources were inspected for the 2026-09-21 extension/review.
The implemented experiment is described in the
[technical summary](design_implementation_summary.md),
[linked methods](two_sided_extension.md), and
[data contracts](data_dictionary.md). This project's own fitted results are
generated separately in [two_sided_results.md](two_sided_results.md).
External methodological examples are not evidence of company policies or
substitutes for those generated results.

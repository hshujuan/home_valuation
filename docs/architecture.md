# Repository design

## Purpose and scope

This repository is an educational, synthetic-data laboratory for three related
problems:

1. estimating a home's value and uncertainty;
2. estimating how acquisition offers and resale prices causally affect
   completion; and
3. choosing supported actions using contribution economics rather than
   conversion alone.

It contains two intentionally separate studies:

- the **baseline study** examines valuation, seller acceptance, resale
  repricing, instrumental-variable diagnostics, and one-stage policy
  optimization; and
- the **linked two-sided study** follows prospects through acquisition
  escalation, inventory, resale markdown, and final disposition.

The package is the single implementation of the models. The command-line
interface, Streamlit app, examples, generated reports, and tests call into that
package rather than maintaining separate analytical logic.

## Design principles

| Principle | How the repository enforces it |
|---|---|
| Separate prediction from intervention | `valuation.py` predicts value; `causal.py`, `instruments.py`, and `two_sided.py` estimate responses to actions |
| Respect decision time | Comparable sales are strictly prior, stage features must predate intervention, and immature outcomes are rejected |
| Preserve experimental support | Optimizers choose only among tested and conditionally feasible actions |
| Account for the complete path | Unsold inventory retains terminal value, and lifecycle costs are charged once |
| Keep truth out of models | Structural truth is returned in separate tables and used only for simulator evaluation |
| Separate learning from evaluation | Chronological splits and cohorts prevent final-policy evaluation from training on future outcomes |
| Reproduce artifacts | Seeds, configuration, package versions, source hashes, metadata, and output hashes are recorded |
| Fail explicitly | Invalid action support, timing, maturity, probability, and accounting inputs raise errors instead of producing fallback results |

## System context

```text
                         User entry points
              +---------------+----------------+
              |               |                |
             CLI          Streamlit         Examples
              |               |                |
              +---------------+----------------+
                              |
                    Shared Python package
                              |
       +----------------------+----------------------+
       |                      |                      |
  Synthetic data       Models and estimators    Decision economics
  and oracle truth     and diagnostics          and policy evaluation
       |                      |                      |
       +----------------------+----------------------+
                              |
              CSV, JSON, Markdown, and figures
                              |
                   Tests and report builder
```

The application has no database or external service dependency. Study data is
generated in memory and optionally persisted below `outputs\`. Public Markdown
and figures below `docs\` are updated only when the CLI is run with
`--write-report`.

## Repository boundaries

| Path | Responsibility |
|---|---|
| [`src/home_valuation/`](../src/home_valuation/) | Simulation, feature contracts, estimation, economics, orchestration, and artifact writing |
| [`app/streamlit_app.py`](../app/streamlit_app.py) | Cached, local interactive presentation of the shared studies |
| [`examples/`](../examples/) | Executable walkthroughs that call package functions or inspect generated outputs |
| [`tests/`](../tests/) | Behavioral checks for timing, support, estimators, economics, artifacts, app navigation, and documentation |
| [`scripts/build_pdf_report.py`](../scripts/build_pdf_report.py) | Optional local conversion and validation of the maintained technical summary |
| [`docs/`](.) | Maintained design, contracts, sources, generated result snapshots, and figures |
| `outputs\reference` | Generated baseline datasets, metadata, manifests, truth, reports, and figures |
| `outputs\two_sided` | Generated linked-study datasets, metadata, manifests, truth, reports, and figures |

Generated output directories are not source inputs and are not required in a
fresh checkout.

## Package component map

| Module | Main responsibility | Depends on |
|---|---|---|
| [`simulation.py`](../src/home_valuation/simulation.py) | Baseline properties, market sales, seller offers, resale assignments, IV fixtures, and separate oracle truth | `features.py` |
| [`features.py`](../src/home_valuation/features.py) | Explicit feature allowlists, preprocessing, prior-sale comparable features, action transforms, and maturity filtering | scikit-learn |
| [`valuation.py`](../src/home_valuation/valuation.py) | Chronological model selection, temporal evaluation, smearing correction, and split-conformal intervals | `features.py` |
| [`seller.py`](../src/home_valuation/seller.py) | Acquisition ledger, seller-response model, acquisition curves, and homeowner profiles | shared feature preprocessing |
| [`causal.py`](../src/home_valuation/causal.py) | Difference-in-means, cross-fitted doubly robust estimation, balance checks, and multi-action response models | `features.py` |
| [`instruments.py`](../src/home_valuation/instruments.py) | IV estimation, weak-IV confidence sets, assignment diagnostics, and exclusion sensitivity | statsmodels and linearmodels |
| [`optimization.py`](../src/home_valuation/optimization.py) | Continuation-aware profit matrices, supported choices, off-policy scores, paired comparisons, and portfolio frontiers | response estimates and logged propensities |
| [`evaluation.py`](../src/home_valuation/evaluation.py) | Calibration, bootstrap, Monte Carlo, valuation slices, stress tests, and ablations | fitted package models |
| [`pipeline.py`](../src/home_valuation/pipeline.py) | Baseline orchestration, study container, metadata, tables, reports, and manifests | all baseline modules |
| [`two_sided_simulation.py`](../src/home_valuation/two_sided_simulation.py) | Linked chronological cohorts, feasible stage actions, assignment, outcomes, and lifecycle truth | baseline valuation and simulation |
| [`two_sided.py`](../src/home_valuation/two_sided.py) | Stage models, continuation rewards, joint acquisition values, sequential DR evaluation, and linked artifacts | `two_sided_simulation.py` and shared helpers |
| [`visualization.py`](../src/home_valuation/visualization.py) | Matplotlib figures used by generated documentation | pipeline result tables |
| [`__main__.py`](../src/home_valuation/__main__.py) | Argument validation and dispatch to the two study orchestrators | `pipeline.py`, `two_sided.py` |

Dependencies flow from generators and feature contracts toward estimators,
decision logic, and orchestration. Presentation code depends on the
orchestrators; package modules do not depend on the app or examples.

## Baseline study flow

The `pipeline` command calls `run_study()` and follows this sequence:

```text
market_sales
    |
    v
fit_valuation -----> temporal metrics, predictions, conformal interval
    |
    +------> resale_population
    |            |
    |            +--> randomized / observed / hidden binary logs
    |            |        +--> difference in means, AIPW, balance
    |            |
    |            +--> randomized multi-action log
    |                     +--> chronological train/test split
    |                     +--> causal response model
    |                     +--> profit and policy evaluation
    |                     +--> stress curves and portfolio frontier
    |
    +------> seller_offers
    |            +--> seller response and acquisition curves
    |
    +------> IV fixtures
                 +--> estimates, weak-IV checks, assignment checks,
                      exclusion sensitivity, and Monte Carlo diagnostics
```

Important boundaries in this path:

- `add_comps()` excludes same-day and future transactions.
- AVM model selection uses a tuning period, conformal calibration uses a later
  period, and final metrics use the latest period.
- Binary causal examples deliberately compare randomized, observed-confounded,
  and hidden-confounded assignment.
- The multi-action response model trains before a market-day cutoff and is
  evaluated after it.
- Policy choices are restricted to `-6%`, `-3%`, hold, and `+3%`.
- `profit_matrices()` gives an unsold home a stipulated continuation value
  instead of treating it as zero.
- Evaluator-only truth produces oracle diagnostics but is not passed as a
  model feature.

`run_study()` returns a `Study` containing configuration, result tables,
metadata, fitted models, and truth tables. `save_study()` is the persistence
boundary.

## Linked two-sided study flow

The `two-sided` command calls `run_two_sided()`. It fits a separate AVM and
creates three nonoverlapping chronological cohorts:

```text
development cohort
    +--> fit acquisition-stage completion response
    +--> fit resale-stage completion response

continuation cohort
    +--> construct mature downstream rewards under:
         - hold resale policy
         - optimized resale policy
    +--> fit acquisition continuation-value models

evaluation cohort
    +--> freeze all learned models
    +--> select supported acquisition escalations
    +--> score acquisition and resale paths with sequential DR
    +--> compare fixed policies on the same held-out prospects
```

The development cohort's lifecycle rewards mature before the continuation
cohort starts, and continuation rewards mature before evaluation starts.
`require_maturity()` and explicit cohort date checks enforce that ordering.

At each stage, the feasible action set is determined before assignment.
Logged actions are uniformly randomized within that set, and scoring verifies
both the action and its probability. The acquisition decision compares
predicted lifecycle contribution under a downstream resale rule, rather than
multiplying acquisition probability by an unconditional average margin.

The linked study is deliberately a fixed two-stage prototype: one acquisition
escalation and one resale markdown. It is not a general repeated-decision or
capital-allocation solver.

## Data and artifact contracts

The maintained column-level contract is in the
[data dictionary](data_dictionary.md). At a higher level, data falls into four
categories:

| Category | Examples | Allowed use |
|---|---|---|
| Decision features | property attributes, prior comps, pre-action demand or engagement signals | model training and policy decisions when available before intervention |
| Logged design fields | assigned action, assignment probability, feasibility, decision date | causal estimation and off-policy evaluation |
| Mature outcomes | acceptance, acquisition, completed sale, realized contribution | training or evaluation only after the declared horizon |
| Simulator truth | latent value, hidden demand, oracle response probabilities | diagnostics only; saved under `evaluator_truth` |

Each persisted run contains:

- one CSV per public result table;
- `metadata.json` with configuration, estimands, diagnostics, and limitations;
- `manifest.json` with artifact hashes, source hashes, package versions, and
  schema version;
- an `evaluator_truth\` directory separated from deployable tables; and
- generated Markdown and figures.

The baseline and linked schemas are versioned independently through their
manifests. Consumers should use table names and documented columns rather than
depending on in-memory object layout.

## Execution surfaces

### Command line

[`__main__.py`](../src/home_valuation/__main__.py) validates command-specific
sizes and resampling flags, constructs a configuration, runs one study, and
persists it. The two commands are intentionally independent:

```powershell
python -m home_valuation pipeline
python -m home_valuation two-sided
```

### Streamlit

[`streamlit_app.py`](../app/streamlit_app.py) caches `run_study()` and
`run_two_sided()` by seed and size. Its sections expose package tables and
models but do not refit alternative dashboard-only estimators. App defaults
favor responsiveness and therefore do not reproduce the full reference run.
The [interactive demo guide](app_guide.md) documents each section, its
controls, and the tables it reads.

### Examples and reports

Examples are thin walkthroughs over package behavior. Generated reports format
the same result tables produced by the orchestrators. The optional PDF builder
converts the maintained technical summary; it does not rerun either study.

## Reliability and validation strategy

The implementation prefers invariant checks near the code that owns the
contract:

- configuration classes reject undersized studies;
- feature builders reject nonpositive prices and invalid actions;
- stage models reject unavailable, post-decision, or immature data;
- causal scorers verify support and known assignment probabilities;
- optimizers reject infeasible choices rather than extrapolating;
- persistence records exact source and artifact hashes; and
- documentation tests verify local links and Markdown counterparts for PDFs.

The test suite covers model and policy behavior, causal algebra, temporal
leakage, oracle separation, support, accounting, deterministic generation,
the application, and report assembly. CI runs linting, tests, and compact
versions of both studies on Windows.

## Extending the design

When adding a new model, action, or study stage:

1. define its decision population, action unit, horizon, and estimand;
2. add predecision fields to an explicit feature allowlist;
3. keep latent simulator truth in a separate returned table;
4. record assignment probabilities and conditional feasibility;
5. require outcome maturity before fitting or scoring;
6. update economics so each cash flow is charged exactly once;
7. expose the result through the orchestrator before adding presentation code;
8. add persisted metadata and schema documentation; and
9. test timing, support, invalid inputs, and a known synthetic target.

New presentation surfaces should consume `Study` or `TwoSidedStudy` outputs.
They should not duplicate feature engineering, fitting, or policy selection.

## Current limitations

- All data and behavior are synthetic; results are not externally calibrated.
- Homes are simulated independently, so equilibrium and correlated portfolio
  shocks are not modeled.
- The AVM supplies point predictions and a marginal conformal interval, not a
  joint distribution over value, repairs, timing, and market states.
- The baseline study uses a stipulated terminal disposition rule.
- The linked study fixes initial offers and listings and supports only one
  escalation and one markdown.
- Portfolio optimization demonstrates a completion/contribution frontier but
  does not implement full dynamic capital, concentration, or risk constraints.
- Policy intervals are conditional on frozen learned rules and are not
  adjusted for every model-selection or multiple-comparison choice.

For the modeling rationale and interpretation, see the
[technical design summary](design_implementation_summary.md) and
[detailed analysis](analysis_report.md). The
[linked two-sided methods](two_sided_extension.md) derive the continuation
rewards and sequential policy scores. For exact fields, timing, and units,
see the [data dictionary](data_dictionary.md). For the
interactive surface, see the [interactive demo guide](app_guide.md).

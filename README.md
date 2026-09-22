# Home valuation and causal pricing lab

A reproducible technical study of **home valuation**, **causal price response**,
and **inventory profit optimization**, motivated by questions about Opendoor.
All executable examples use newly generated synthetic data. This is not
Opendoor's production model, proprietary data, or financial advice.

The main deliverable is an interactive Streamlit demo of how causal inference
supports decisions on both sides of an iBuyer marketplace:

- **Acquisition:** valuation uncertainty, homeowner response, offer acceptance,
  adverse selection, and offer policy.
- **Resale:** buyer response, markdown effects, completion probability,
  contribution profit, inventory risk, and price policy.

Start with the [concise technical summary](docs/design_implementation_summary.pdf),
then launch the app or run the reproducible command-line studies.

| Read | Purpose |
|---|---|
| [Concise technical summary](docs/design_implementation_summary.pdf) | Shareable overview of the two-sided valuation and causal decision system |
| [Technical summary source](docs/design_implementation_summary.md) | Maintained Markdown source for the concise PDF |
| [Generated reference results](docs/reference_results.md) | Actual numbers from the seeded Python pipeline |
| [Source ledger](docs/sources.md) | Public sources, supported claims, and verification limits |
| [Data dictionary](docs/data_dictionary.md) | Units, timestamps, labels, costs, and feature restrictions |
| [Generated two-sided results](docs/two_sided_results.md) | Separate linked-cohort results, including uncertainty and unchanged recommendations |

## What the study does

The main experiment compares **holding the current resale list price** with a
**3% price cut**, among homes still unsold at a specified decision time. Its
outcome is a completed sale over the next 30 days. A separate acquisition
example models homeowner beliefs, reservation utility, final-offer acceptance,
completed acquisitions, and selection into the purchased pool.

The reference simulation estimates a roughly **8.95 percentage-point** resale
completion lift using randomized, cross-fitted AIPW. For one illustrated home,
a 3% cut increases predicted completion from about **17.86% to 25.93%**, yet
reduces expected contribution from about **$32,358 to $30,559**. These are
synthetic results, not estimates of any company's real price elasticity.

An **additive linked extension** follows the same prospect from initial offer
through selective escalation, completed acquisition, listing, markdown, and
disposition. Three chronological cohorts separate response development,
downstream-reward learning, and final evaluation. The seed-42 selective rule
holds all acquisition offers; its policy interval does not establish a gain.
That unfavorable result is retained, not tuned away. PDF Parts IX-X cover
this extension without replacing the original experiment.

![Synthetic price response and profit](docs/figures/response_and_profit.png)

## Project structure

| Path | Purpose |
|---|---|
| `src/home_valuation/` | Simulation, valuation, causal estimation, policy evaluation, and optimization |
| `app/streamlit_app.py` | Interactive visual demo for acquisition and resale decisions |
| `examples/` | Small executable walkthroughs of each modeling stage |
| `tests/` | Leakage, causal identification, economics, policy, app, and report checks |
| `docs/` | Technical summary, generated results, sources, data contracts, and figures |
| `scripts/build_pdf_report.py` | Reproducible technical-summary PDF builder |

## Install on Windows

Use Python 3.11 and run commands from this project directory. No activation
script or real-data access is required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

`requirements.txt` pins the tested Windows/Python 3.11 environment, including
demo and development dependencies. To install from the compatible version
ranges declared in `pyproject.toml` instead, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[demo,dev]"
```

The modeling package itself does not require Streamlit; `pip install -e .`
is sufficient for the CLI and examples. The complete test suite includes
the app and therefore needs both extras. Native numerical threads are bounded
in the main study and AVM fitting to avoid excessive CPU oversubscription.

## Launch the interactive demo

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Open **http://127.0.0.1:8501**. The checked-in configuration binds the app to
localhost and disables Streamlit usage telemetry. Stop it with `Ctrl+C`.

## Read or rebuild the technical summary

`docs\design_implementation_summary.pdf` is the concise, standalone overview;
its maintained source is `docs\design_implementation_summary.md`.

To rebuild the summary from the maintained Markdown and figures:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[report]"
.\.venv\Scripts\python.exe scripts\build_pdf_report.py --summary --date 2026-09-21
```

The optional report extra supplies Pandoc, Typst, and PyMuPDF. It is not
needed for modeling or the app.
The builder runs locally without fetching web content, records source hashes
and tool versions in `outputs\pdf_summary\manifest.json`, and checks the PDF
before replacing the published copy. Temporary build files remain ignored.

Rebuilding the summary **does not rerun the simulation**. After changing a model,
first regenerate the affected study, reconcile the narrative's seed-42
numbers, then rebuild. Omitting `--date` uses today's snapshot date; the date
is not a claim that public sources were reverified that day.

## Generate datasets and reproduce the report

```powershell
# Quick run: same models, smaller data, no resampling loops.
.\.venv\Scripts\python.exe -m home_valuation pipeline --size 1000 --bootstrap 0 --monte-carlo 0 --output outputs\smoke

# Reference run: default table sizes, seed 42, uncertainty and IV repetitions.
.\.venv\Scripts\python.exe -m home_valuation pipeline --seed 42 --bootstrap 30 --monte-carlo 30 --write-report --output outputs\reference
```

The reference run creates 5,000 market sales, 4,000 homeowner offers, 6,000
eligible resale decisions per assignment design, and 8,000 observations per
main IV fixture. It writes CSVs, metadata, hashes, evaluator-only truth,
result tables, and figures under `outputs\reference`.
`--write-report` also refreshes `docs\reference_results.md` and `docs\figures`.
It does not overwrite the narrative analysis or source ledger.

Runs with different seeds or sizes will produce different estimates. The
narrative's numeric examples refer to the documented seed-42 reference run.
Thirty bootstrap/Monte Carlo repetitions are intentionally modest teaching
settings, not production-quality tail inference. Longer runs can increase
both flags without changing the generator or selecting favorable seeds.

## Run the eight walkthroughs

Run the reference pipeline first; the scripts read its generated CSV files.

```powershell
.\.venv\Scripts\python.exe examples\01_eda.py
.\.venv\Scripts\python.exe examples\02_features_and_valuation.py
.\.venv\Scripts\python.exe examples\03_homeowner_valuation.py
.\.venv\Scripts\python.exe examples\04_causal_price_response.py
.\.venv\Scripts\python.exe examples\05_instrumental_variables.py
.\.venv\Scripts\python.exe examples\06_profit_optimization.py
.\.venv\Scripts\python.exe examples\07_stress_and_policy_validation.py
.\.venv\Scripts\python.exe examples\08_design_diagnostics.py
```

These are readable Python entrypoints, not eight independent implementations.
They call the shared package in `src\home_valuation` or inspect the pipeline's
generated tables; they do not maintain duplicate model logic or result files.
Start with EDA, then
valuation, homeowner decisions, causal response, instruments, optimization,
then stress/offline policy evaluation and the new instrument/portfolio diagnostics.

The comparison-driven upgrade adds conditional IV exclusion sensitivity,
observed assignment checks, paired policy contrasts, and a supported-price
portfolio completion frontier. A fractional optimizer result is a lottery
over tested prices, not an unsupported average price.

## Run the linked two-sided extension and ninth walkthrough

```powershell
.\.venv\Scripts\python.exe -m home_valuation two-sided --seed 42 --write-report
.\.venv\Scripts\python.exe examples\09_linked_two_sided.py
```

This separate command defaults to `outputs\two_sided`, with 5,000 historical
transactions and 12,000/10,000/8,000 prospects in development, continuation,
and evaluation cohorts. It writes linked datasets, stage effects, candidate
values, chosen-action gains, and sequential DR policy comparisons.
`--write-report` updates only `docs\two_sided_results.md` and its figure.
For a small run, use `--size 2400 --output outputs\two_sided_smoke`.
Bootstrap/Monte Carlo flags apply only to the original `pipeline` command.

The [generated two-sided results](docs/two_sided_results.md) report the linked
funnel, stage effects, downstream policy values, and final policy evaluation.
The implementation is in `src\home_valuation\two_sided.py` and
`src\home_valuation\two_sided_simulation.py`.

## Explore the interactive demo

The eight sections show data, valuation errors, homeowner profiles, causal
estimates, valid and invalid IV designs, price/profit curves, and assumptions.
In **Profit optimization**, change the home, objective, terminal-value
assumption, holding costs, or price floor. An impossible floor produces an
explicit abstention rather than an invented feasible price.
The instrument and profit sections also expose the new diagnostics and
portfolio frontier. That frontier uses the study's fixed holdout portfolio and
reference economics; it does not change with the individual-home stress sliders.
**Two-sided extension** adds linked funnels, stage effects, candidate acquisition
values, chosen actions, and joint policy uncertainty using the same package.

The app defaults to 3,000 observations per table and omits resampling loops
for responsiveness. It therefore does **not** reproduce the full reference
run's exact numbers. The CLI-generated report includes bootstrap summaries;
the app displays the same core estimators, economic stresses, and policy
comparisons at its selected configuration.
The linked section uses `max(3600, 2 * selected size)` prospects per cohort,
not the default extension CLI cohort sizes; its configuration is displayed.

## Checks

```powershell
.\.venv\Scripts\python.exe -m ruff check src examples app tests scripts
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover temporal leakage, oracle separation, treatment units, outcome
maturity, cost accounting, support, infeasibility, IV confidence-set inversion,
selection derivatives, stress-policy choices, deterministic simulation, and
application navigation, plus portfolio feasibility/shadow prices, stochastic
policy scores, paired differences, and exclusion sensitivity. With the report
extra installed, additional checks cover PDF source assembly, table panels,
and internal links. The GitHub Actions workflow repeats the core checks and compact runs of both
experiments on Windows; it does not require the optional
PDF toolchain. Linked-stage checks include sequential-score algebra, actual
chosen-gain hurdles, whole-cohort accounting, and downstream-policy linkage.

## Privacy and scope

The repository contains the maintained package, app, examples, tests, build
scripts, public documentation, and selected synthetic figures. Virtual
environments, caches, generated `outputs`, raw data, private reference
material, notebooks, and comparison-only implementations are excluded by
`.gitignore` and are not required to run the project. The generator uses only
synthetic data and does not read external or private inputs.

No push or deployment is part of running any example. Public research
citations are links, not redistributed copies of papers. The project does
not claim legal compliance, production calibration, or externally valid
pricing recommendations.

# Narrative Macro-at-Risk thesis repository

This MSc thesis repository uses IMF Growth-at-Risk methods to study downside
GDP-growth and upside inflation risks under uncertain macroeconomic
conditions. The active analysis consists of six independent narrative
single-equation models: three narratives estimated for GDP growth and for
inflation. The former baseline Growth-at-Risk model has been archived and is
not part of the current thesis workflow.

## Repository map

| Location | Purpose |
|---|---|
| `config.py` | Active data, six-model registry, quantile, EMA, and output settings. |
| `run_narrative_gar.py` | Main entry point to estimate all six models or a selected subset. |
| `GAR/dnb_gar.py` | Thesis-specific loading, target alignment, shared model adapter, diagnostics, and outputs. |
| `GAR/quantfit/`, `GAR/tsfit/` | IMF quantile-regression and conditional T-skew implementations used by the adapter. |
| `analysis/run_ema_sampling.py` | Growth-only EMA sampling for the three narrative specifications. |
| `tests/` | Configuration, alignment, fitting, and output tests. |
| `BQR data yoy.xlsx` | Prepared quarterly data, worksheet `data_yoy`. |
| `results/narrative_models/` | Results for each narrative/outcome pair. |
| `results/ema_sampling/` | Narrative EMA experiment outputs. |
| `Documentation/`, `GaR license.txt`, `IMF disclaimer.txt` | Original IMF documentation and attribution. |
| `archive/` | Preserved baseline, legacy Excel workflow, presentation materials, and an old environment snapshot; not used by active scripts. |

Original IMF components remain in `GAR/`; `GAR/dnb_gar.py`, the root runner,
configuration, and EMA script are thesis-specific adapters/workflows. The
baseline runner, its old outputs, and baseline-dependent presentation
materials have been moved to `archive/`.

## Data and active model specifications

The scripts load `BQR data yoy.xlsx`, worksheet `data_yoy`, using
`pandas.read_excel`. `NF_YER` is treated as year-on-year real GDP growth.
`NF_HIC` is the HICP-related inflation outcome and is used exactly as
provided. The available documentation does not establish whether `NF_HIC`
is specifically a year-on-year rate or fully describe its transformation;
no additional transformation is applied. The BMPE variable list provides
underlying economic concept definitions, but not a complete record of each
prepared BQR column's transformation and units.

Each narrative uses the same predictors for both outcomes:

| Narrative | Predictors |
|---|---|
| Financial and Credit Conditions | `NF_YER`, `DF_LTN_NFCN`, `DF_STN_NFCN`, `DF_STO` |
| External Demand and Competitiveness | `NF_YER`, `DN_CXD`, `DR_FOD` |
| Household Demand and Housing Market | `NF_YER`, `DR_PYR`, `DF_LTN_HHHN`, `DF_IHX` |

The external-demand specification excludes `DR_WDR`. `NF_YER` remains a
predictor in both outcome families. Growth's primary risk measure is the
conditional 5th percentile (downside Growth-at-Risk); inflation's is the
conditional 95th percentile (upside Inflation-at-Risk). Direct regression
and fitted T-skew quantiles at all seven configured levels are saved.

Edit [`config.py`](config.py) to change model predictors, outcomes, the
four-quarter horizon, quantiles, EMA sample count, random seed, sampling
method, or input-range percentiles. The three narrative specifications are
defined once and shared across outcomes.

## Run

Run commands in a VS Code PowerShell terminal from the repository root.
Create an environment and install the project runtime/test dependencies if
needed:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scipy statsmodels matplotlib openpyxl ema-workbench pytest
```

Estimate all six models, one outcome family, or a specific combination:

```powershell
python run_narrative_gar.py
python run_narrative_gar.py --outcome growth
python run_narrative_gar.py --outcome inflation
python run_narrative_gar.py --model model_1_financial_conditions --outcome inflation
python run_narrative_gar.py --model model_2_external_demand --outcome growth
```

Run EMA for all six models, one outcome family, one narrative, or one
narrative/outcome pair:

```powershell
python analysis/run_ema_sampling.py
python analysis/run_ema_sampling.py --outcome growth
python analysis/run_ema_sampling.py --outcome inflation
python analysis/run_ema_sampling.py --model model_1_financial_conditions
python analysis/run_ema_sampling.py --model model_1_financial_conditions --outcome inflation
python -m pytest -q
```

The runners accept `--output-dir` to direct results elsewhere. Narrative fits
are written to `results/narrative_models/<narrative>/<growth|inflation>/`.
EMA outputs are separate under
`results/ema_sampling/<narrative>/<growth|inflation>/` and do not overwrite
historical model-fit results.

## Outputs and interpretation

Each narrative/outcome folder contains quantile-regression coefficients and
diagnostics, quantile-crossing diagnostics, `model_summary.json`, in-sample
conditional quantiles, `latest_forecast.csv`, and outcome-labelled figures.
The model summary records sample count and forecast/target date spans.
In-sample fitted values use the full estimation sample and are not historical
real-time forecasts. The latest forecast uses only outcomes available at its
information cutoff.

For each narrative, EMA draws one Latin Hypercube design and evaluates both
the growth and inflation equations at each identical input combination.
Each scenario has a stable input-derived `scenario_id`, so results are matched
by `narrative_id` and `scenario_id`, not by row order. Both outcomes use
provisional 5th-to-95th percentile bounds computed from the full available
history of each narrative predictor. Thus the same bounds apply despite
outcome-specific differences in aligned estimation samples. Coefficients are
estimated once per model and remain fixed during sampling.

Each outcome folder contains all seven direct quantile-regression and fitted
T-skew quantiles, fitted parameters, regression and scenario quantile-crossing
diagnostics, failure records, metadata, a reference forecast, summaries, and
outcome-labelled figures. The narrative folder also contains
`combined_scenarios.csv`; by default it includes both conditional marginal
distributions evaluated under the same scenarios. If EMA is run for one
outcome, the table is updated to include only that selected outcome, as
recorded in `outcomes_included`. It is **not** a joint distribution: the
analysis does not assume independence, estimate joint probabilities, or
calculate stagflation probabilities. Scenario frequency is design coverage,
not the probability of future events.

## Method and limitations

The adapter aligns each origin-date predictor row with the selected outcome
four quarters later and uses the existing IMF quantile-regression and
conditional T-skew fitting functions. It reports direct quantile estimates
and T-skew distribution quantiles separately. See
[`METHODOLOGY.md`](METHODOLOGY.md) for estimation and validation details.

Model histories differ because some predictors have shorter coverage.
`NF_HIC`'s exact transformation is uncertain. Quantile crossing can occur in
the unconstrained direct quantile regressions and is reported in diagnostics.
EMA bounds are provisional and independent sampling does not preserve
historical input dependencies. Comprehensive out-of-sample validation,
PRIM, joint densities, copulas, and full robustness analysis are not
implemented.

## IMF attribution

The original IMF GaR Python/Excel implementation and documentation are
preserved. The active pipeline reuses IMF estimation and distribution-fitting
code; the data adapter, six narrative specifications, runners, EMA workflow,
and thesis documentation are thesis-specific. Retain
[`GaR license.txt`](GaR%20license.txt) and
[`IMF disclaimer.txt`](IMF%20disclaimer.txt) with redistributed IMF material.
Reuse does not imply IMF endorsement.

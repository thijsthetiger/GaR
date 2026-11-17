# Narrative Growth-at-Risk and Inflation-at-Risk methodology

## Scope and implementation

The active thesis workflow estimates six separate single-equation models:
three economic narratives for GDP growth and the same three for inflation.
The previously used baseline Growth-at-Risk specification has been archived
and is excluded from active estimation and EMA. No narrative is treated as a
replacement for that baseline.

The thesis-specific adapter in `GAR/dnb_gar.py` loads and aligns data, applies
the information cutoff, calls the IMF quantile-regression and conditional
T-skew functions, and writes model outputs. The estimation procedures are
shared across all six models. IMF statistical functions are not redefined by
the narrative workflow. Active specifications and numerical settings are
centralized in `config.py`.

## Data and specifications

The input is `BQR data yoy.xlsx`, worksheet `data_yoy`. The data loader
validates quarter-end dates, ordering, duplicates, consecutive quarters,
required columns, and numeric values. `NF_YER` is the GDP-growth outcome.
`NF_HIC` is the HICP-related inflation outcome and is used without an
additional transformation. Its observed values appear consistent with an
inflation rate rather than a positive index level, but the available
workbooks do not establish that it is specifically a year-on-year rate or
document its exact transformation. This uncertainty is not resolved by
silently transforming the series.

The three predictor sets are configured once in `config.py` and used
unchanged with either dependent variable:

| Narrative | Predictors |
|---|---|
| Financial and Credit Conditions | `NF_YER`, `DF_LTN_NFCN`, `DF_STN_NFCN`, `DF_STO` |
| External Demand and Competitiveness | `NF_YER`, `DN_CXD`, `DR_FOD` |
| Household Demand and Housing Market | `NF_YER`, `DR_PYR`, `DF_LTN_HHHN`, `DF_IHX` |

`DR_WDR` is excluded from the external-demand specification, while `DR_FOD`
is retained. `NF_YER` remains a predictor in inflation models as well as
growth models. The configurations are six independent equations, not a
joint growth-inflation distribution.

Some predictors have shorter histories, so the complete estimation samples
differ between narratives. Output `model_summary.json` records the available
observations and date ranges for each outcome-specific fit.

## Outcome alignment and estimation

For forecast origin quarter \(t\), the adapter aligns the chosen outcome at
\(t+4\) with predictors observed at \(t\). The forecast horizon is four
quarters, and target dates are stored separately from origin dates. At an
information cutoff \(T\), a training row is retained only if its target date
is no later than \(T\), preventing use of unavailable future outcomes.

The IMF `GAR.quantfit.quantilereg.QuantileReg` implementation fits the
configured conditional quantiles (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
and an OLS conditional mean. The adapter passes the configured maximum
iterations, parameter tolerance, and confidence-interval alpha. Iteration
limit warnings become explicit errors; non-finite coefficients and invalid
training data are also rejected. Quantile crossings are diagnosed in the
direct regression outputs and reported rather than silently corrected.

The fitted conditional quantiles are passed to the IMF
`GAR.tsfit.tskewfit.tskew_fit` function. The resulting conditional
distribution parameters and quantiles are validated for finite values,
positive scale/skew, and strictly increasing distribution quantiles. The
T-skew fitter does not expose its optimizer success flag, so these checks do
not constitute a formal convergence diagnostic.

## Tail-risk interpretation

For GDP growth, the primary indicator is the fitted conditional 5th
percentile, interpreted as downside Growth-at-Risk. For inflation, the
primary indicator is the fitted conditional 95th percentile, interpreted as
upside Inflation-at-Risk. The direct quantile-regression estimate and
parametric T-skew quantile are both saved; they can differ because the
T-skew fit represents the entire quantile set. All seven direct and fitted
quantiles and distribution parameters are retained for further analysis.

Historical fitted conditional quantiles use a model estimated on the
available sample and are in-sample, not real-time historical forecasts. The
latest forecast uses latest origin-date inputs and estimates only with target
outcomes available by the information cutoff.

## EMA and limitations

`analysis/run_ema_sampling.py` runs preliminary EMA input-sampling
experiments for all six narrative models. For each narrative, it creates one
Latin Hypercube design and evaluates the growth and inflation equations at
the same sampled predictor combinations. Scenario IDs are stable hashes of
the narrative and exact sampled values; downstream tables are matched by
`narrative_id` and `scenario_id`, not by row order or Workbench scenario
numbers.

The shared empirical input ranges use the configured lower and upper
percentiles of each predictor's full available history in the raw BQR data.
This intentionally avoids using outcome-specific aligned estimation
samples to set bounds, so both outcomes receive identical bounds even where
their estimation histories differ. Ranges remain provisional, independent
marginal percentiles; independent sampling does not preserve historical
dependencies. Each equation is fitted once before sampling, and its
regression coefficients remain fixed while the conditional quantiles and
T-skew distribution are evaluated at each scenario.

The primary Growth-at-Risk indicator is the fitted conditional 5th
percentile; the primary Inflation-at-Risk indicator is the fitted
conditional 95th percentile. All seven direct and fitted quantiles and the
distribution parameters are written to each outcome's scenario table. A
per-narrative `combined_scenarios.csv` joins the two outcome-specific
conditional marginal results using the shared scenario identifiers. It is
not a joint density or probability distribution: no dependence assumption,
joint probability, or stagflation probability is estimated. Scenario
frequencies describe coverage of the exploratory design, not probabilities
of future events. Regression convergence information and scenario-level
direct-quantile crossing counts are saved as diagnostics. Select
`--outcome growth` or `--outcome inflation` to run a single outcome family;
in that case, the combined table contains only the selected outcome, as
recorded in its `outcomes_included` column.

The baseline model and baseline-dependent presentation workflow are
preserved under `archive/` but are not part of active research. Comprehensive
out-of-sample backtesting, PRIM, joint densities, copulas, and full
robustness analysis are not implemented. Results are conditional
associations and should not be interpreted causally. Prepared BQR column
transformations and units, including the exact `NF_HIC` transformation, need
further documentation.

## IMF attribution

The original IMF Growth-at-Risk package and documents remain available in
`GAR/` and `Documentation/`; license and disclaimer files are retained at
the repository root and in `GAR/`. The thesis-specific adapter, narrative
specifications, runners, EMA integration, and documentation reuse the IMF
implementation but do not imply IMF endorsement.

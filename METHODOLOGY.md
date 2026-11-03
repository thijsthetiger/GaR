# Thesis methodology: IMF-based DNB Growth-at-Risk

## Research purpose and boundary

The first implementation estimates how DNB macroeconomic and financial inputs
relate to the conditional distribution of Dutch real GDP growth four quarters
ahead. It is a transparent baseline for later Exploratory Modeling and
Analysis (EMA), not an EMA application. PRIM, a joint input density, and
Inflation-at-Risk are outside the current scope.

## Source data and preparation

`BQR data yoy.xlsx`, worksheet `data_yoy`, contains 221 quarterly observations
from 1971Q1 through 2026Q1. Dates are unique, consecutive, and quarter-end
dates. `NF_YER` is the dependent variable; it is treated as year-on-year real
GDP growth as checked by the thesis owner. The official variable list is the
`List_variables` worksheet (capitalization as stored) of
`BMPE_vintages_QLY (1).xlsx`.

The listed official concepts are:

- `YER`: GDP - Chain linked volume (`NF_YER`).
- `URX`: Unemployment rate (% of labour force) (`NF_URX`).
- `LTN_NFCN`: long term lending rate for NFC (`DF_LTN_NFCN`).
- `PYR`: Gross Personal Disposable Income, Real (deflated by consumption
  deflator) (`DR_PYR`).

The `NF_`, `DF_`, and other BQR column prefixes are retained exactly as stored.
The available workbooks do not provide a complete variable-by-variable account
of how `data_yoy` was transformed from the official underlying series or the
units after each transformation. The model therefore uses the chosen predictor
values as supplied, does not transform them again, and reports their
coefficients per BQR-column unit. The missing transformation details are an
explicit data limitation, not an assumption about the meaning of the observed
values.

The current specification uses `DR_PYR` instead of `DF_IHX`. This is a
coverage-motivated substitution: `DR_PYR` is non-missing for all 221 workbook
quarters, whereas `DF_IHX` first appears in 1999Q1 and its use left 105
complete origin/target pairs. With `DR_PYR`, the four-quarter target
requirement leaves 217 pairs. This does not establish that disposable-income
growth is theoretically superior to house prices.

## Predictor specification

The initial four-predictor specification is theory-led:

1. `NF_YER`, current output growth, captures persistence in real activity.
2. `NF_URX`, unemployment, captures labour-market slack.
3. `DF_LTN_NFCN`, the BQR series based on the long-term lending rate for
   non-financial corporations, proxies the cost of corporate finance.
4. `DR_PYR`, the BQR series based on real gross personal disposable income,
   represents household income conditions. The substitution was made to
   extend historical coverage; no causal or theoretical claim is implied.

Adrian, Boyarchenko, and Giannone (2019) motivate the financial-conditions
channel: tighter financial conditions are associated with weaker lower
quantiles of future GDP growth. The current model also includes a real
disposable-income series, but the variable substitution is motivated by
coverage rather than a newly established causal channel. No expected
coefficient sign is imposed in estimation.

The unemployment-growth link is consistent with the output-unemployment
relationship summarized by Ball, Leigh, and Loungani (2013). For `NF_YER`,
current GDP growth is included as a persistence/current-conditions predictor
within the IMF conditional-growth framework; this is a parsimonious modeling
choice rather than a causal claim.

## Outcome and forecast horizon

For an observation at quarter \(t\), the model target is
\[
    y_{t+4} = \texttt{NF\_YER}_{t+4},
\]
and the predictors are observed at \(t\). This is the year-on-year growth rate
four quarters later, not a sum of the next four quarterly growth rates. The
adapter constructs the target by shifting the outcome series backward four
rows while retaining the predictor date as the forecast origin and storing
the outcome date separately.

Before fitting at information cutoff \(T\), the adapter retains only training
rows with target dates at or before \(T\). Thus a latest-origin forecast at
2026Q1 uses target observations only through 2026Q1; the 2027Q1 outcome is not
available to estimation.

## Quantile regression

The IMF `GAR.quantfit.quantilereg.QuantileReg` class estimates conditional
quantile regressions at 5%, 10%, 25%, 50%, 75%, 90%, and 95%, along with an OLS
conditional-mean regression. Regressions use the supplied predictor units
(`scaling=False`) so coefficients can be reported in their BQR-column units.
The IMF class uses Statsmodels' formula-based quantile regression, a maximum
of 2,000 iterations, and parameter tolerance `1e-5`. The adapter turns an
iteration-limit warning into an explicit model error and checks the resulting
coefficients for finite values.

The IMF package received compatibility-only changes: Python's removed `imp`
module was replaced with `importlib`, Excel/Windows imports are optional for
non-Excel use, deprecated DataFrame `_append` calls in the used quantile
regression module were replaced with `pandas.concat`, the scaler is imported
only when scaling is requested, and an unconditional T-skew diagnostic print
was removed. The quantile-regression and T-skew estimation methods themselves
were not changed.

## Conditional distribution and Growth-at-Risk

For a predictor scenario, the seven fitted conditional quantiles are passed to
the IMF `GAR.tsfit.tskewfit.tskew_fit` function. This preserves the IMF
T-skew parameterization and its scale/skew optimization. The resulting
conditional distribution is evaluated with the IMF `tskew_ppf`, `tskew_cdf`,
and `tskew_pdf` functions.

The primary distribution-based GaR estimate is the conditional 5th percentile
of this fitted T-skew distribution. The direct 5th-percentile quantile
regression is also output as a distinct, non-parametric tail estimate. These
two estimates need not coincide because the T-skew is a parametric fit to the
entire set of conditional quantiles.

## In-sample results versus forecast

The historical figure and `in_sample_conditional_quantiles.csv` show fitted
values from a model estimated on the available sample. They are explicitly
labelled in-sample: at historical dates, their coefficients use observations
that would not have been known in real time. `latest_forecast.csv` is separate
and uses the latest input values at the information cutoff, with target date
four quarters later.

In the updated run, 217 paired observations are used, with origin dates from
1971Q1 through 2025Q1. At the 2026Q1 origin, the target is 2027Q1; the direct
quantile-regression 5th percentile is −2.433% and the fitted T-skew 5th
percentile is −2.549%, with a fitted 14.3% chance of negative growth. These
conditional associations are not causal estimates, and the predictor
transformations require confirmation before economic interpretation.

No claim of improved forecast performance is made. The repository did not
contain a previous thesis expanding-window backtest implementation, and this
initial baseline does not claim out-of-sample predictive superiority.

## Limitations and next validation

- The exact transformations and units of some prepared BQR input columns are
  not documented in either supplied workbook. Confirm these with the data
  owner before interpreting coefficient magnitudes or passing alternative
  scenarios.
- The change to `DR_PYR` increases the sample, but 5% tail estimates remain
  statistically uncertain and the longer historical sample may span changing
  economic regimes.
- The IMF T-skew fitting function returns fitted parameters but does not expose
  the underlying optimizer's success flag. The adapter verifies finite,
  positive distribution parameters, ordered quantiles, and CDF/quantile
  consistency; these checks do not replace a convergence-status diagnostic.
- Linear quantile regressions do not impose coefficient signs or nonlinear
  interactions. The T-skew fit adds a distributional shape assumption.
- A later step should add a strictly expanding-window backtest with
  origin-specific cutoffs, proper tail scoring, and parameter stability
  diagnostics before making forecast-performance claims.
- For EMA, expose the predictor values as uncertainties/levers and call the
  already-separated `predict` method. EMA inputs must respect the same
  transformations and units as the BQR columns; these transformations need
  to be resolved first.

## References and attribution

- Adrian, T., Boyarchenko, N., & Giannone, D. (2019). “Vulnerable Growth.”
  *American Economic Review*, 109(4), 1263–1289.
  https://doi.org/10.1257/aer.20161923
- Ball, L., Leigh, D., & Loungani, P. (2013). “Okun's Law: Fit at Fifty?”
  NBER Working Paper 18668. https://doi.org/10.3386/w18668
- Iacoviello, M. (2005). “House Prices, Borrowing Constraints, and Monetary
  Policy in the Business Cycle.” *American Economic Review*, 95(3), 739–764.
  https://doi.org/10.1257/0002828054201477
- Mian, A., Sufi, A., & Verner, E. (2015). “Household Debt and Business Cycles
  Worldwide.” NBER Working Paper 21581. https://doi.org/10.3386/w21581
- IMF, *Growth at Risk* Python implementation in `GAR/`, version 0.1.1,
  dated in the repository README as 28 October 2019. The source is governed by
  CC BY-NC-SA 4.0; see `GAR/GaR license.txt`. The IMF disclaimer in
  `IMF disclaimer.txt` applies. Reuse does not imply IMF endorsement.

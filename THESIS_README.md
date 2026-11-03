# DNB Growth-at-Risk thesis model

This repository contains the original IMF Growth-at-Risk (GaR) Python
implementation and a small DNB-specific, non-Excel entry point. The thesis
adapter reuses the IMF quantile-regression estimator and T-skew distribution
fit. It does not implement the EMA Workbench, PRIM, a joint density, or
Inflation-at-Risk.

## Run the model

Use Python with `pandas`, `openpyxl`, `numpy`, `statsmodels`, `scipy`, and
`matplotlib` installed. scikit-learn is only needed if the original IMF
quantile-regression class is used with standardization enabled; this model
keeps predictor units unscaled:

```powershell
python run_dnb_gar.py
```

By default the command reads `BQR data yoy.xlsx`, worksheet `data_yoy`, and
writes results to `results/`. Paths can be specified explicitly:

```powershell
python run_dnb_gar.py --data "BQR data yoy.xlsx" --output-dir results
```

The entry point does not open Excel or modify either workbook. The existing
IMF Excel workflow remains available when its `xlwings` and Windows `pywin32`
dependencies are installed.

## Variables and rationale

| BQR column | Definition from `BMPE_vintages_QLY (1).xlsx`, `List_variables` | Role and expected relationship | Literature basis | Transformation used here |
|---|---|---|---|---|
| `NF_YER` | `YER`: GDP - Chain linked volume | Current year-on-year GDP growth captures output persistence and current macroeconomic conditions. Weak current growth can be associated with weaker growth four quarters ahead. | The IMF GaR framework models the conditional distribution of future GDP growth given current conditions; Adrian et al. (2019). | Used as reported in `data_yoy`; the thesis owner confirms this series is year-on-year real GDP growth. |
| `NF_URX` | `URX`: Unemployment rate (% of labour force) | Current labour-market slack is a macroeconomic condition; higher unemployment is expected to be associated with weaker subsequent growth. | Ball, Leigh, and Loungani (2013), cross-country evidence on the output-unemployment relationship. | Used as reported in `data_yoy`; no additional transformation is applied. |
| `DF_LTN_NFCN` | `LTN_NFCN`: long term lending rate for NFC | Corporate borrowing costs proxy financial conditions. Tighter/more expensive credit conditions are expected to be associated with lower future growth and greater downside risk. | Adrian et al. (2019), financial conditions and lower GDP-growth quantiles. | Used as reported in `data_yoy`; the workbook does not specify the exact transformation or transformed unit for this column. |
| `DR_PYR` | `PYR`: Gross Personal Disposable Income, Real (deflated by consumption deflator) | Real disposable income growth represents household income conditions. It replaces `DF_IHX` to use the longer available history, not because this substitution establishes a superior theoretical specification. | Coverage-motivated variable choice; the current model does not impose a causal interpretation. | Used as reported in `data_yoy`; the workbook does not specify the exact transformation or transformed unit for this BQR column. |

These are a theory-led initial specification, not variables selected to maximize
historical fit. Adrian, Boyarchenko, and Giannone (2019) document that tighter
financial conditions are associated with worsening lower quantiles of GDP
growth. Mian, Sufi, and Verner (2015) show that household debt growth predicts
weaker future growth internationally; that evidence motivates attention to
financial-cycle vulnerabilities but does not make the selected income series
a substitute for a direct credit/debt measure.

The variable list defines the underlying economic concepts. `data_yoy` is a
prepared BQR sheet, not a raw-data sheet; neither workbook documents a
column-by-column transformation recipe or transformed units. Accordingly, the
adapter preserves the four BQR input series as supplied and makes no further
transformation. Outcome figures are labelled in the NF_YER year-on-year growth
units; predictor coefficients are per one unit of the relevant BQR column.
`DR_PYR` replaces `DF_IHX`, which begins in 1999Q1, because `DR_PYR` is
available from 1971Q1 in the supplied BQR workbook. The model still uses the
same four-quarter horizon and IMF quantile-regression and T-skew fitting
methods. With this coverage change, the complete paired sample increases from
105 to 217 observations.

## What is produced

- `quantile_regression_coefficients.csv`: raw-scale IMF quantile-regression
  coefficients, p-values, confidence intervals, and in-sample fit statistics.
- `quantile_regression_diagnostics.csv`: estimation sample sizes and available
  iteration counts for each fitted quantile.
- `in_sample_conditional_quantiles.csv`: full-sample conditional quantiles,
  T-skew quantiles, realized future outcomes, and distribution parameters.
  These are **in-sample fitted values**, not genuine historical forecasts.
- `latest_forecast.csv`: a separate prediction at the latest observed input
  date, with its target date, raw 5th-percentile quantile-regression estimate,
  IMF T-skew 5th-percentile Growth-at-Risk estimate, and other quantiles.
- `historical_growth_at_risk.png`: in-sample fitted GaR and the aligned
  four-quarter-ahead realizations.
- `latest_conditional_distribution.png`: latest conditional IMF T-skew density
  and its lower 5% tail.

The CSV coefficient file includes all seven conditional quantile regressions
and an OLS conditional-mean row. The main distribution-based GaR measure is the
IMF T-skew 5th percentile; the direct 5th-percentile quantile regression is
reported separately so the parametric fit is transparent.

For the supplied workbook, the updated estimation contains 217 complete
paired observations, with forecast origins from 1971Q1 through 2025Q1. The
latest forecast origin is 2026Q1 and its target is 2027Q1. The direct
quantile-regression 5th percentile is **−2.433%**; the IMF T-skew 5th
percentile is **−2.549%**, with an estimated 14.3% probability of growth below
zero. These are model-based estimates, not realized outcomes or evidence of
out-of-sample forecast performance. Do not interpret the estimated
associations causally, especially while transformed units remain undocumented.

## Python interface and future EMA use

`GAR.dnb_gar.DNBGrowthAtRisk` separates estimation from prediction:

```python
from GAR.dnb_gar import (
    DNBGrowthAtRisk,
    load_dnb_data,
    prepare_horizon_data,
)

data = load_dnb_data("BQR data yoy.xlsx")
aligned = prepare_horizon_data(data, horizon=4)
model = DNBGrowthAtRisk(horizon=4).fit(aligned, as_of=data.index.max())
result = model.predict(
    {
        "NF_YER": 0.58,
        "NF_URX": 6.26,
        "DF_LTN_NFCN": -0.01,
        "DR_PYR": 1.00,
    }
)
```

For use as an EMA model, the future adapter can pass an input mapping to
`predict` and collect `distribution_quantiles[0.05]`,
`quantile_regression[0.05]`, or the full distribution parameters and quantiles.
No EMA Workbench integration is included yet.

## Attribution and license

The IMF source code is governed by the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International Public License included
in `GAR/GaR license.txt`. The IMF quantile-regression and T-skew fitting
algorithms are reused without algorithmic changes. Compatibility-only changes
were made to the IMF package entry/import path, the quantile-regression
DataFrame concatenation and optional scaler import, and a module-level T-skew
debug print. `GAR/dnb_gar.py`, `run_dnb_gar.py`, and this thesis documentation
are DNB/thesis-specific adaptations. Retain the IMF attribution, license, and
disclaimer when sharing adapted material. Reuse does not imply IMF endorsement.

## References

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

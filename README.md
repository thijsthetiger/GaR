# DNB Growth-at-Risk thesis repository

This repository contains the International Monetary Fund's (IMF) Growth-at-Risk
(GaR) implementation and a thesis-specific adapter that applies its
quantile-regression and T-skew distribution methods to prepared Dutch
macroeconomic data from De Nederlandsche Bank (DNB). It is the methodological
starting point for an MSc Engineering & Policy Analysis thesis at TU Delft.

The research asks whether Exploratory Modeling and Analysis (EMA) can improve
the robustness and policy relevance of insights from Macro-at-Risk (MaR)
models. The current repository provides an initial DNB GaR baseline; it does
**not** implement EMA, scenario discovery, PRIM, or a robustness evaluation.
The existing model and its calculations are intentionally kept separate from
those future research activities.

## Repository structure

The current layout is retained deliberately. The DNB runner resolves its
default workbook relative to the repository root, and the adapter imports the
IMF modules through the existing `GAR` package. Moving these files would
require code/path changes, which are outside the scope of this documentation
and organization pass.

```text
.
├── README.md                         This repository guide
├── METHODOLOGY.md                    Existing detailed model/data methodology
├── THESIS_README.md                  Existing DNB model run and variable notes
├── BQR data yoy.xlsx                 Prepared DNB input workbook
├── BMPE_vintages_QLY (1).xlsx        Variable-definition/context workbook
├── run_dnb_gar.py                    Current standalone DNB model entry point
├── run_GAR.py                       Legacy IMF/Excel-oriented wrapper
├── gar.xlsm                         Original IMF Excel tool/workbook
├── GAR/
│   ├── dnb_gar.py                   Thesis-specific data/model/output adapter
│   ├── __init__.py, globals.py      IMF package and Excel integration
│   ├── quantfit/                    IMF quantile-regression implementation
│   ├── tsfit/                       IMF T-skew fitting and distribution code
│   ├── historical/                  IMF historical analysis modules
│   ├── partition/                   IMF partition analysis modules
│   ├── scenario/                    IMF scenario analysis modules
│   └── segment/                     IMF segment analysis modules
├── Documentation/                   IMF GaR paper, technical docs, and license
├── GaR license.txt                  License copy
├── IMF disclaimer.txt               IMF disclaimer
├── readme.txt                       Original/legacy IMF package instructions
├── python_env.txt                   Existing environment information
├── results/                         Existing generated CSV and PNG outputs
└── tests/
    └── test_dnb_gar.py              Tests for the thesis-specific adapter
```

`.venv/` is a local Python environment, not part of the source tree or model
inputs. It is excluded from Git by `.gitignore`.

## Research context and model boundary

GaR summarizes downside risk to future economic growth using a conditional
distribution rather than only a conditional mean. The IMF implementation
estimates conditional quantiles and fits a parametric T-skew distribution to
them; its lower tail can then be reported as Growth-at-Risk. This DNB baseline
uses the prepared BQR worksheet as the source of model observations.

The current implementation is a baseline to understand and evaluate before
extending it. The model estimates associations, not causal effects. No claim
that it improves forecast performance, or that EMA has already improved its
robustness or policy relevance, is made here. See [METHODOLOGY.md](METHODOLOGY.md)
for the detailed specification, alignment choices, assumptions, and current
limitations.

## IMF source and thesis-specific contributions

### IMF material

The original IMF Python package is the source code under `GAR/`, **except**
`GAR/dnb_gar.py`, which is the thesis-specific adapter. The IMF source modules
are organized in these existing package paths:

- `GAR/__init__.py` and `GAR/globals.py`
- `GAR/historical/`: `__init__.py`, `asymt.py`, `asymtfit.py`,
  `historical_gen.py`, `historicalmain.py`, `tskew.py`, `tskewfit.py`
- `GAR/partition/`: `__init__.py`, `PLSDA.py`, `cpdgrowth.py`,
  `gen_cutoff.py`, `partition.py`, `partition_cutoff.py`,
  `partition_main.py`, `partition_retro.py`, `plot_partition.py`,
  `retropolate_func.py`
- `GAR/quantfit/`: `__init__.py`, `condqgreg.py`, `plot_quantfit.py`,
  `quantfit.py`, `quantfit_main.py`, `quantilereg.py`
- `GAR/scenario/`: `__init__.py`, `asymt.py`, `asymtfit.py`, `relation.py`,
  `scenario_compare.py`, `scenariomain.py`, `tskew.py`, `tskewfit.py`
- `GAR/segment/`: `__init__.py`, `PLSDA.py`, `asymt.py`, `asymtfit.py`,
  `condqgreg.py`, `cpdgrowth.py`, `gen_cutoff.py`, `gen_seg_skewt.py`,
  `partition.py`, `partition_cutoff.py`, `partition_retro.py`,
  `plot_partition.py`, `plot_quantfit.py`, `quantfit.py`, `quantilereg.py`,
  `relation.py`, `retropolate_func.py`, `scenario_compare.py`,
  `segmentmain.py`, `term_plot.py`, `tskew.py`, `tskewfit.py`
- `GAR/tsfit/`: `__init__.py`, `asymt.py`, `asymtfit.py`, `gen_skewt.py`,
  `tsfit_main.py`, `tskew.py`, `tskewfit.py`

The original IMF Excel-facing files and reference material are also retained:
`run_GAR.py`, `gar.xlsm`, `readme.txt`, the license and disclaimer files, and
the material under `Documentation/`. They are not the entry point for the
current standalone DNB run. Keep IMF attribution, license, and disclaimer
material with any redistribution; reuse does not imply IMF endorsement.

**Pre-existing compatibility edits:** this documentation pass did not edit
Python source. Before this pass, the worktree already contained local changes
to `GAR/__init__.py`, `GAR/quantfit/quantilereg.py`, and `GAR/tsfit/tskew.py`.
As described in the existing methodology, these changes concern optional
Excel/Windows imports and compatibility/debug details, not a change to the
quantile-regression or T-skew estimation methods. They remain in place and
should be reviewed separately against the IMF distribution before treating
those files as pristine upstream copies.

### Thesis-specific files

- `GAR/dnb_gar.py` loads and validates the BQR data, aligns the future target
  with each forecast origin, fits through the IMF quantile-regression class,
  calls the IMF T-skew fit and distribution functions, and writes tables and
  figures.
- `run_dnb_gar.py` is the current command-line entry point. It connects the
  workbook, adapter, model fit, and output writing.
- `tests/test_dnb_gar.py` contains focused tests for horizon alignment,
  information-cutoff filtering, finite prediction output, T-skew quantile/CDF
  consistency, and invalid input handling.
- `METHODOLOGY.md` and `THESIS_README.md` are thesis/DNB-specific documentation.

The DNB adapter remains inside `GAR/` because the runner and adapter use
package imports such as `GAR.tsfit...`; relocating it into a new `thesis_model/`
directory would require changing Python imports. Likewise, the workbooks
remain in the root because the current default input path is there. No files
were moved in this pass.

## Current data and variable specification

| Item | Verified current setting |
|---|---|
| Model input workbook | `BQR data yoy.xlsx` |
| Worksheet loaded by code | `data_yoy` (the adapter's default sheet) |
| Dependent variable / outcome series | `NF_YER` |
| Predictor columns | `NF_YER`, `NF_URX`, `DF_LTN_NFCN`, `DR_PYR` |
| Quantile-regression quantiles | 5%, 10%, 25%, 50%, 75%, 90%, and 95%, plus an OLS conditional-mean fit |
| Forecast horizon | Four quarters |
| Horizon target | `NF_YER` observed four quarterly rows after the predictor date |
| BMPE workbook | `BMPE_vintages_QLY (1).xlsx`; documentation source for variable concepts |
| BMPE read by model code? | No; the code references and reads the BQR workbook, not the BMPE workbook |

`NF_YER` serves both as the current-growth predictor and the series whose
four-quarter-ahead value is forecast. The adapter does not apply additional
transformations to the selected BQR columns. The exact preparation,
transformations, and units of some BQR columns are not fully documented by the
workbooks; `NF_YER` is described in the existing methodology as year-on-year
real GDP growth. Do not infer that the BQR column names alone establish their
transformation or units.

The existing methodology identifies the BMPE workbook's `List_variables`
worksheet as the source of the official variable definitions. It lists `PYR`
as real gross personal disposable income; `DR_PYR` is the selected BQR series,
described here as real disposable income growth. The BMPE file provides
contextual definitions; no direct data/code connection to it is implemented.
The BQR sheet, not BMPE, supplies model observations.

`DR_PYR` replaces `DF_IHX` because it is available for all 221 workbook
quarters, while `DF_IHX` begins in 1999Q1 and leaves only 105 four-quarter
origin/target pairs. With `DR_PYR`, the same horizon and information cutoff
produce 217 pairs. This is a data-coverage reason for the substitution, not
evidence that the new predictor is methodologically superior. The adapter
does not add transformations to any selected BQR series; the exact upstream
preparation and units should still be confirmed before interpreting
coefficients.

## Current workflow

1. `run_dnb_gar.py` defaults to the root-level `BQR data yoy.xlsx` workbook.
   The `--data` option can select another workbook; the adapter reads its
   `data_yoy` sheet by default.
2. `GAR.dnb_gar.load_dnb_data` parses the first worksheet column as dates,
   checks quarterly date ordering/completeness, verifies required columns, and
   converts model columns to numeric values.
3. `prepare_horizon_data` keeps the predictors at their origin date and aligns
   `NF_YER` four quarters later as the future target. It also retains the
   corresponding target date.
4. The runner fits `DNBGrowthAtRisk` using the latest observed input date as
   the information cutoff. Training rows whose target date is later than that
   cutoff are excluded.
5. The adapter calls the IMF `QuantileReg` implementation for the seven
   conditional quantiles and OLS conditional mean. It uses `scaling=False`,
   so inputs are not standardized by the regression class.
6. For predictions, the fitted conditional quantiles are passed to IMF
   `tskew_fit`; the T-skew functions then supply distribution quantiles and
   the probability of negative growth. The reported distribution-based GaR is
   the fitted T-skew 5th percentile. The direct 5th-percentile regression is
   reported separately.
7. `write_model_results` writes the coefficient and diagnostic tables,
   in-sample fitted conditional quantiles, the latest-origin forecast, and two
   figures to the output directory.

The historical fitted values use a model estimated on the available training
sample and are **in-sample**, not historical real-time forecasts. The separate
latest forecast uses the latest input values and reports a target date four
quarters later. The existing outputs and the timing distinction are discussed
in [METHODOLOGY.md](METHODOLOGY.md).

## Running and testing

Use a Python environment containing `pandas`, `openpyxl`, `numpy`,
`statsmodels`, `scipy`, and `matplotlib` for the standalone model. The adapter
sets `scaling=False`, so scikit-learn is not needed for this DNB path. Running
the tests additionally requires `pytest`. `python_env.txt` is an existing
package inventory, not an installable requirements file.

Run the current standalone DNB model from the repository root:

```powershell
python run_dnb_gar.py
```

The default input is `BQR data yoy.xlsx` and the default output directory is
`results/`. To specify either path:

```powershell
python run_dnb_gar.py --data "BQR data yoy.xlsx" --output-dir results
```

The command writes or overwrites the generated files listed below. Do not run
it if you need to preserve the existing output snapshot unchanged; use a
different `--output-dir` to retain that snapshot. The test command runs the
repository's focused adapter tests without reading the data workbook or
writing model-result files:

```powershell
python -m pytest tests/test_dnb_gar.py
```

The updated specification was verified with the existing adapter tests
(`5 passed`) and a run on the supplied workbook (`217` estimation
observations). The run produced quantile-regression and T-skew forecast
outputs; see `results/latest_forecast.csv`.

`run_GAR.py` is the legacy IMF wrapper, not the DNB entry point. Its current
main block uses the IMF Excel path and calls the segment operation; other
stages in its `main` function are commented out. The original workbook-driven
workflow requires the applicable Excel automation dependencies (including
`xlwings` and Windows `pywin32`) and is distinct from `run_dnb_gar.py`.

## Important files

| File/folder | Purpose | Needed for current DNB run? | Guidance |
|---|---|---:|---|
| `BQR data yoy.xlsx` | Prepared model input, `data_yoy` worksheet | Yes | Supply the input data; do not edit without documenting provenance and transformations |
| `BMPE_vintages_QLY (1).xlsx` | Variable definitions/context | No, not read by code | Consult for definitions; retain as documentation input |
| `run_dnb_gar.py` | Standalone command-line runner | Yes | Use to run the current model |
| `GAR/dnb_gar.py` | DNB-specific data adapter, model wrapper, and output writer | Yes | Thesis-specific; modify only in a separately scoped model-development step |
| `GAR/quantfit/quantilereg.py` | IMF conditional quantile regression | Yes | IMF-derived; leave unchanged unless making an explicit, reviewed source change |
| `GAR/tsfit/tskewfit.py`, `GAR/tsfit/tskew.py` | IMF T-skew estimation and distribution calculations | Yes | IMF-derived; leave unchanged |
| Other modules under `GAR/` | IMF Excel, historical, partition, scenario, and segment functionality | No for the standalone DNB path, except dependencies named above | Retain; consult for original IMF functionality |
| `tests/test_dnb_gar.py` | Focused adapter behavior tests | No at model runtime | Run after code changes; do not treat as full scientific validation |
| `results/` | Existing generated CSV tables and figures | No as input | Treat as outputs/snapshots; runner may overwrite files |
| `METHODOLOGY.md` | Detailed DNB specification and limitations | No | Consult and keep aligned with any future approved model changes |
| `THESIS_README.md` | Existing DNB run, variable, and result notes | No | Consult; this README is the main repository guide |
| `run_GAR.py`, `gar.xlsm`, `readme.txt` | Original IMF Excel-related workflow | No for DNB runner | Retain for provenance; do not confuse with the standalone DNB command |
| `Documentation/`, license/disclaimer files | IMF papers, technical material, usage guidance, and attribution | No at runtime | Retain and consult for source provenance and attribution |
| `python_env.txt` | Existing environment information | No at runtime | Consult as local environment documentation; it is not the DNB entry point |
| `.venv/` | Local virtual environment | No as repository content | Kept locally and ignored by Git |

The original IMF subpackages include additional modules that are not imported
by the standalone DNB workflow. Their presence does not indicate that every
IMF feature (for example, IMF Excel, partition, historical, scenario, or
segment workflows) is part of the current thesis model.

## Results currently present

The existing `results/` directory contains:

- `quantile_regression_coefficients.csv`
- `quantile_regression_diagnostics.csv`
- `in_sample_conditional_quantiles.csv`
- `latest_forecast.csv`
- `historical_growth_at_risk.png`
- `latest_conditional_distribution.png`

These are generated artifacts, not source data. The coefficient table includes
quantile regressions and the OLS mean fit. The diagnostics table records
available sample-size and iteration information. The in-sample table contains
fitted conditional quantiles and aligned outcomes; the latest forecast is a
separate forecast-origin result. Existing files are retained and are not
recomputed by the documentation update.

## Validation status and limitations

The existing tests exercise date/horizon alignment, information-cutoff
filtering, finite predictions, T-skew quantile/CDF consistency, and selected
input validation. They are implementation checks, not a full replication of
the IMF Excel workflow, an independent methodological validation, or an
out-of-sample forecast evaluation.

The current supplied data provide 217 complete paired observations with
`DR_PYR`. The latest origin is 2026Q1 and target is 2027Q1. These are
snapshot-specific outputs, not guarantees for a changed workbook or future
rerun.

Important limitations documented for the present baseline include the
incomplete record of some BQR transformations and units, uncertainty in
estimated tail quantiles with a limited sample, and the absence of an
expanding-window backtest with appropriate tail scoring and stability
diagnostics. The conditional coefficients should not be interpreted
causally. The IMF T-skew fit checks described in the methodology do not
substitute for an optimizer convergence-status diagnostic.

## Future development (not implemented)

An eventual EMA extension could expose the current predictor values as
uncertainties or policy levers, define plausible ranges or scenarios, and call
the adapter's separated prediction interface to evaluate conditional tail
outputs. Before doing so, the BQR transformations and units must be resolved
so scenario values are comparable to the model inputs. Scenario analysis,
PRIM (Patient Rule Induction Method), and robustness evaluation could then
identify input combinations associated with adverse outcomes and assess how
conclusions vary across assumptions. None of these extensions is present in
the current code; no EMA or PRIM functionality is implied by the IMF
`scenario` package.

## Attribution and references

The IMF GaR Python implementation is version 0.1.1 and the original README
dates it to 28 October 2019. See the included GaR license and IMF disclaimer.
The thesis-specific DNB adapter and its documentation should remain clearly
distinguished from that source.

Selected methodological references are listed in [METHODOLOGY.md](METHODOLOGY.md),
alongside the model assumptions, variable rationale, and data caveats.

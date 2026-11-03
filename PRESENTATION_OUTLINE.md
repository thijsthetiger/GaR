# Presentation outline: Exploring Input Uncertainty in Macro-at-Risk Models

**Audience:** MSc Engineering & Policy Analysis thesis supervisors, TU Delft / DNB  
**Length:** Approximately 8 slides  
**Evidence boundary:** Model specification, coefficients, forecasts, and figures below are read from the current repository and existing output files. EMA, PRIM, and sampled scenario results are proposed future work; none is implemented or reported as an empirical finding here.

## Slide 1 — Exploring Input Uncertainty in Macro-at-Risk Models

**Key message:** The thesis will investigate whether EMA can show how uncertainty in macroeconomic inputs changes MaR/GaR tail-risk insights.

**Slide content**

- **Research objective:** Investigate whether Exploratory Modeling and Analysis (EMA) can improve the robustness and policy relevance of insights from Macro-at-Risk (MaR) models.
- MaR focuses on the conditional distribution and adverse tail of macroeconomic outcomes, not only on an average forecast.
- GaR is a GDP-focused example: it summarizes downside risks to future growth conditional on current macro-financial conditions.
- Inputs are uncertain, measured imperfectly, and may vary across plausible futures. A forecast conditional on one input vector does not show how tail risk changes across those alternatives.
- **Current status:** An IMF-based DNB GDP Growth-at-Risk implementation is the initial model for investigating the EMA connection. EMA itself is not implemented.

**Figure / visual**

- Simple research flow: `DNB macroeconomic inputs → conditional GDP-growth distribution → downside tail measure`.
- Label the current GaR implementation “baseline model”; label EMA “proposed extension.”

**Speaker explanation**

The main thesis question is not simply whether one model produces the most accurate point forecast. It is whether plausible variation in the model inputs can materially alter conclusions about downside macroeconomic risk, and whether an EMA approach can make those vulnerabilities more transparent and policy-relevant. The current model supplies a concrete conditional distribution model to build on.

**Sources**

- `README.md` — research context and current/future functionality boundary.
- `METHODOLOGY.md` — current GaR purpose and scope.

## Slide 2 — Baseline Growth-at-Risk Model

**Key message:** The current model estimates four-quarter-ahead Dutch GDP growth conditional on four contemporaneous BQR series.

**Slide content**

| Model element | Current specification |
|---|---|
| Data input | `BQR data yoy.xlsx`, worksheet `data_yoy` |
| Sample in the current fitted results | 217 origin–target pairs |
| Forecast origins in estimation | 1971Q1–2025Q1 |
| Target dates in estimation | 1972Q1–2026Q1 |
| Forecast target | `NF_YER` four quarters after the predictor date |
| Forecast horizon | Four quarters |
| Conditional quantiles | 5%, 10%, 25%, 50%, 75%, 90%, 95% |

| Predictor | Economic interpretation | Data caveat |
|---|---|---|
| `NF_YER` | Current GDP growth; also the series being forecast | Existing documentation treats it as year-on-year real GDP growth |
| `NF_URX` | Unemployment rate | BQR values are used as supplied |
| `DF_LTN_NFCN` | BQR series associated with the long-term lending rate for NFCs | Exact BQR transformation/units are not fully documented |
| `DR_PYR` | BQR series associated with real gross personal disposable income (`PYR` in the BMPE variable list); used as disposable-income growth in this specification | Exact BQR transformation/units are not fully documented |

The current model replaced `DF_IHX` with `DR_PYR` to extend historical coverage. All four selected series have 221 non-missing values in the input sheet; the four-quarter target alignment leaves 217 estimation pairs. The substitution is a coverage choice, not proof that `DR_PYR` is a theoretically superior predictor.

**Figure / visual**

- Draw a timeline: predictors dated at \(t\), `NF_YER` target dated at \(t+4\).
- Put the 217-observation count and sample dates beside the timeline.

**Speaker explanation**

Be explicit that `NF_YER` appears in two roles: its value at the forecast origin is one of the predictors, while its value four quarters later is the outcome. The code shifts the series by four quarterly rows. The model uses the data already prepared in the BQR workbook; it does not read or transform data from the BMPE workbook. The BMPE variable list is context for definitions, not a model input.

**Sources**

- `GAR/dnb_gar.py` — `PREDICTORS`, `DEPENDENT_VARIABLE`, `DEFAULT_HORIZON`, `DATA_SHEET`, `load_dnb_data`, `prepare_horizon_data`.
- `run_dnb_gar.py` — default BQR path, estimation, and information cutoff.
- `BQR data yoy.xlsx` — `data_yoy` sheet and sample coverage.
- `BMPE_vintages_QLY (1).xlsx` — `List_variables` sheet, including the `PYR` definition.
- `results/quantile_regression_diagnostics.csv` — seven fitted quantile regressions, each with 217 observations.
- `results/in_sample_conditional_quantiles.csv` — origin and target date range.

## Slide 3 — Mathematical Formulation of the GaR Model

**Key message:** The model first estimates conditional quantiles with separate linear regressions, then fits an IMF T-skew distribution to those quantile estimates.

**Slide content**

Let \(Y_{t+4}=\texttt{NF\_YER}_{t+4}\) be the target and let
\[
X_t =
\left(
\texttt{NF\_YER}_t,\,
\texttt{NF\_URX}_t,\,
\texttt{DF\_LTN\_NFCN}_t,\,
\texttt{DR\_PYR}_t
\right).
\]
For each fitted quantile \(\tau\in\{0.05,0.10,0.25,0.50,0.75,0.90,0.95\}\), the quantile regression is
\[
Q_{\tau}(Y_{t+4}\mid X_t)
=\beta_{0,\tau}
+\beta_{1,\tau}\texttt{NF\_YER}_t
+\beta_{2,\tau}\texttt{NF\_URX}_t
+\beta_{3,\tau}\texttt{DF\_LTN\_NFCN}_t
+\beta_{4,\tau}\texttt{DR\_PYR}_t.
\]

For \(\tau=0.05\), the current fitted coefficients are:

| Term | Coefficient |
|---|---:|
| Intercept | −5.855645 |
| `NF_YER` at \(t\) | 0.295874 |
| `NF_URX` at \(t\) | 0.458065 |
| `DF_LTN_NFCN` at \(t\) | −0.787512 |
| `DR_PYR` at \(t\) | 0.371629 |

Therefore, the actual fitted direct 5th-percentile regression is
\[
\begin{aligned}
Q_{0.05}(Y_{t+4}\mid X_t)
={}&-5.855645
+0.295874\,\texttt{NF\_YER}_t
+0.458065\,\texttt{NF\_URX}_t\\
&-0.787512\,\texttt{DF\_LTN\_NFCN}_t
+0.371629\,\texttt{DR\_PYR}_t.
\end{aligned}
\]

- \(\tau\) is the conditional quantile level; \(\tau=0.05\) denotes the 5th percentile.
- Each \(\beta_{j,\tau}\) is estimated separately for each quantile, so relationships can differ across the conditional distribution. These are conditional associations, not causal effects.
- At a given \(\tau\), each slope describes the fitted conditional quantile's change with that predictor, holding the other included predictors fixed.
- Inputs are passed with `scaling=False`. Coefficients are in the units of the provided BQR columns; the exact transformations/units of some columns are not established in the workbook documentation.
- The seven predicted quantiles are then supplied to the IMF T-skew fitting function. Write the fitted distribution quantile as
  \[
  Q_{\tau}^{T}(X_t)=F^{-1}_{T}\!\left(\tau;\widehat{loc}(X_t),\widehat{scale}(X_t),\widehat{skew}(X_t),\widehat{df}(X_t)\right).
  \]
  The reported distribution-based GaR is \(Q_{0.05}^{T}(X_t)\). The direct QR 5th percentile and fitted T-skew 5th percentile are separate outputs and need not be equal.
- The probability of negative growth is
  \[
  \Pr(Y_{t+4}<0\mid X_t)=F_T(0\mid X_t).
  \]

**Figure / visual**

- Use the coefficient table above, emphasizing the fifth-percentile equation.
- Optionally add a two-box schematic: “Seven QR estimates” → “IMF T-skew fit” → “T-skew 5th percentile / probability below zero.”

**Speaker explanation**

The displayed numerical equation is the direct \(\tau=0.05\) regression from the saved coefficient output; it is not the final T-skew distribution equation. For each input vector, the model evaluates all seven fitted quantile regressions. The T-skew fitting step approximates those quantiles with a single parametric conditional distribution. This distinction explains why the direct QR lower-tail estimate and the distribution-based GaR can differ.

**Sources**

- `GAR/dnb_gar.py` — `QUANTILES`, `DNBGrowthAtRisk.fit`, `DNBGrowthAtRisk.predict`.
- `GAR/quantfit/quantilereg.py` — IMF `QuantileReg` formula and quantile fitting.
- `GAR/tsfit/tskewfit.py` — IMF `tskew_fit`.
- `GAR/tsfit/tskew.py` — IMF `tskew_ppf` and `tskew_cdf`.
- `results/quantile_regression_coefficients.csv` — all numerical coefficients.

## Slide 4 — Conditional Growth Distribution and Tail Risks

**Key message:** The latest model run places the fitted 5th-percentile growth estimate below zero, while its in-sample history is not a real-time forecast evaluation.

**Slide content**

For the latest input origin, 2026Q1, with target date 2027Q1:

| Output | Current result |
|---|---:|
| Direct QR 5th percentile | −2.433% |
| Fitted T-skew 5th percentile (reported GaR) | −2.549% |
| Direct QR median, \(\tau=0.50\) | 2.098% |
| Fitted T-skew median | 2.153% |
| Fitted probability of growth below zero | 14.3% |
| T-skew parameters | \(loc=2.858,\ scale=0.9003,\ skew=0.6042,\ df=2\) |

- The 5th percentile is a lower-tail cutoff of the fitted conditional distribution; it is not the probability of negative growth.
- The 14.3% estimate is the T-skew CDF evaluated at zero.
- In `latest_conditional_distribution.png`, the red dashed line marks T-skew GaR and the shaded area is the lower 5% tail.
- `historical_growth_at_risk.png` shows full-sample in-sample fitted GaR alongside the aligned realized future growth values. These historical fitted values use the full-sample estimated relationships; they are not expanding-window or real-time forecasts.
- No out-of-sample forecast performance claim is supported by these outputs.

**Figures**

1. `results/latest_conditional_distribution.png` — primary figure; annotate the 5th percentile and negative-growth threshold if needed.
2. `results/historical_growth_at_risk.png` — secondary figure; retain its “in-sample fitted” label.

**Speaker explanation**

The conditional distribution answers “what growth distribution does the model associate with these inputs?” Its 5th percentile is the reported downside-tail measure, while the CDF at zero answers a different question: the model-implied chance of negative growth. The historical chart is descriptive in-sample fit. It should not be read as evidence that the model predicted each historical outcome in real time.

**Sources**

- `results/latest_forecast.csv` — current origin, target, quantiles, T-skew parameters, and negative-growth probability.
- `results/latest_conditional_distribution.png`.
- `results/historical_growth_at_risk.png`.
- `GAR/dnb_gar.py` — `write_model_results`, including in-sample status and latest forecast output.

## Slide 5 — From Data to Tail-Risk Forecasts

**Key message:** A small thesis-specific adapter prepares and aligns DNB inputs, then calls IMF quantile-regression and T-skew routines.

**Slide content**

### 1. Load and select BQR data — thesis-specific adapter

Path: `GAR/dnb_gar.py`, constants and `load_dnb_data`.

```python
PREDICTORS = ("NF_YER", "NF_URX", "DF_LTN_NFCN", "DR_PYR")
DATA_SHEET = "data_yoy"
data = pd.read_excel(workbook_path, sheet_name=sheet_name)
```

The runner defaults to `BQR data yoy.xlsx`; the adapter uses the `data_yoy` sheet and validates quarterly dates and required columns.

### 2. Construct the four-quarter-ahead target — thesis-specific adapter

Path: `GAR/dnb_gar.py`, `prepare_horizon_data`.

```python
prepared[TARGET_COLUMN] = data[dependent_variable].shift(-horizon)
prepared[TARGET_DATE_COLUMN] = pd.Series(
    data.index, index=data.index
).shift(-horizon)
prepared = prepared.dropna(
    subset=[*predictors, TARGET_COLUMN, TARGET_DATE_COLUMN]
)
```

Predictors keep their date \(t\); the shifted `NF_YER` is the outcome at \(t+4\). The complete-case selection requires every selected predictor and target.

### 3. Estimate conditional quantile regressions — adapter calls IMF implementation

Path: `GAR/dnb_gar.py`, `DNBGrowthAtRisk.fit`; implementation in `GAR/quantfit/quantilereg.py`, `QuantileReg.__qfit_dict`.

```python
regression = QuantileReg(
    TARGET_COLUMN,
    indvars=list(self.predictors),
    quantile_list=list(self.quantiles),
    data=training,
    scaling=False,
    alpha=0.1,
)
```

The adapter passes the target, predictors, training rows, and seven quantiles to the IMF estimator without standardizing the BQR columns.

### 4. Fit the T-skew distribution — adapter calls IMF implementation

Path: `GAR/dnb_gar.py`, `DNBGrowthAtRisk.predict`; fitting routine in `GAR/tsfit/tskewfit.py`.

```python
distribution = tskew_fit(
    quantile_values.copy(),
    _TSKEW_FIT_PARAMETERS,
)
```

The conditional QR values at the requested input vector are fitted with the IMF T-skew distribution. The adapter then evaluates its quantile and CDF functions.

### 5. Extract the tail-risk output — thesis-specific result writer

Path: `GAR/dnb_gar.py`, `write_model_results`.

```python
"qr_q05": latest_prediction["quantile_regression"][0.05],
"tskew_q05_GaR": latest_prediction["distribution_quantiles"][0.05],
"probability_growth_below_zero": latest_prediction[
    "probability_growth_below_zero"
],
```

The CSV keeps the direct QR 5th percentile distinct from the T-skew 5th percentile and negative-growth probability.

**Figure / visual**

- Use a compact left-to-right pipeline:
  `BQR worksheet → validate/align target → IMF QuantileReg → IMF T-skew fit → latest_forecast.csv + figures`.
- Label the adapter steps “thesis-specific” and the regression/T-skew methods “IMF implementation.”

**Speaker explanation**

The adapter manages DNB data and timing; it does not replace the IMF estimation method. The forecasting origin, target alignment, input selection, and outputs are thesis-specific wiring. The quantile regression and T-skew fitting functions are called from the existing IMF package.

**Sources**

- `run_dnb_gar.py` — entry point and default workbook path.
- `GAR/dnb_gar.py` — `load_dnb_data`, `prepare_horizon_data`, `DNBGrowthAtRisk.fit`, `DNBGrowthAtRisk.predict`, `write_model_results`.
- `GAR/quantfit/quantilereg.py` — IMF regression estimator.
- `GAR/tsfit/tskewfit.py` and `GAR/tsfit/tskew.py` — IMF distribution fitting and functions.
- `results/latest_forecast.csv`.

## Slide 6 — From a Deterministic Forecast to Input Uncertainty

**Key message:** The existing fitted model can be evaluated at multiple input vectors to study conditional tail-risk sensitivity while holding the estimated model fixed.

**Slide content**

**Current deterministic use**

```text
One input vector X_t
        ↓
Fixed estimated QR coefficients and fitted T-skew method
        ↓
One conditional distribution and one set of tail-risk outputs
```

**Proposed EMA use**

```text
Many plausible input vectors X^(1), X^(2), ..., X^(N)
        ↓
Same fixed estimated model for every vector
        ↓
Distribution and tail-risk outputs for every scenario
```

- In an initial experiment, the estimated QR coefficients and T-skew fitting method/settings remain fixed after baseline estimation. For each scenario, the model recomputes its conditional QR values and fits the T-skew distribution to those scenario-specific values; the distribution parameters and risk outputs can therefore change across scenarios.
- The sampled macroeconomic inputs vary between scenarios.
- Record how the T-skew 5th percentile, direct QR 5th percentile, and probability of negative growth vary.
- This is **input uncertainty**. It is not coefficient uncertainty, model-form uncertainty, or stochastic shocks.
- At present, `DNBGrowthAtRisk.predict` accepts one mapping of predictor values and returns conditional quantiles and distribution outputs. There is no EMA Workbench integration or batch scenario runner.

**Figure / visual**

- Show the two flow diagrams side by side, with a bracket around the “fixed model” box in the EMA flow.
- Emphasize that the target of investigation is the variation in conditional downside risk across inputs.

**Speaker explanation**

The current prediction interface already separates fitting from prediction. In principle, one fitted model can be evaluated repeatedly at different input mappings. That provides a plausible point of connection for EMA, but a proper experiment still needs a defensible definition of uncertain inputs, plausible ranges, dependencies, and output measures.

**Sources**

- `GAR/dnb_gar.py` — `DNBGrowthAtRisk.fit` and `DNBGrowthAtRisk.predict`.
- `README.md` — explicitly states EMA/PRIM are not implemented.

## Slide 7 — Exploring Input Uncertainty Through Scenario Sampling

**Key message:** A first EMA design could sample a reviewed range for each input, evaluate every vector with the same fitted model, and compare the resulting tail-risk outputs.

**Slide content**

### Proposed, not yet run

1. Confirm which inputs should be treated as uncertain and whether all four belong in the experiment.
2. As a transparent **provisional** range design, consider each input’s observed 5th–95th percentile in `data_yoy`. The current empirical bounds are:

   | Input | Observed 5th percentile | Observed 95th percentile |
   |---|---:|---:|
   | `NF_YER` | −1.341 | 4.957 |
   | `NF_URX` | 1.979 | 11.300 |
   | `DF_LTN_NFCN` | −1.668 | 1.871 |
   | `DR_PYR` | −1.046 | 4.654 |

   Values are in the prepared BQR column units. These marginal ranges are a starting point for supervisor review, not validated economic plausibility bounds; transformation metadata should be confirmed first.
3. Generate an experimental design such as Latin Hypercube Sampling (LHS), initially specifying whether inputs are sampled independently. Consider preserving observed dependencies if independent combinations are implausible.
4. Pass each sampled input vector through the same fitted `DNBGrowthAtRisk` model.
5. Store scenario inputs plus direct QR \(q_{0.05}\), T-skew \(q_{0.05}\), and \(\Pr(Y_{t+4}<0)\).
6. Define an adverse-outcome threshold before examining scenarios; later use PRIM to identify interpretable input combinations associated with adverse tail-risk outcomes.

### Illustrative table — hypothetical design points, not sampled or evaluated

The input rows below combine observed marginal percentiles only to illustrate the table structure. They are not verified joint scenarios. No GaR has been calculated for them.

| Scenario | Sampled GDP growth `NF_YER` | Sampled unemployment `NF_URX` | Sampled lending rate `DF_LTN_NFCN` | Sampled disposable income growth `DR_PYR` | Resulting T-skew \(q_{0.05}\) GaR |
|---|---:|---:|---:|---:|---|
| Current-input reference (observed 2026Q1) | 0.584 | 6.263 | −0.012 | 1.000 | **−2.549% (existing model forecast)** |
| Illustrative low-side corner (marginal 5th percentiles) | −1.341 | 1.979 | −1.668 | −1.046 | Not evaluated |
| Illustrative high-side corner (marginal 95th percentiles) | 4.957 | 11.300 | 1.871 | 4.654 | Not evaluated |

**Proposed visualizations**

- Histogram of T-skew GaR across the sampled cases.
- Scatterplots of each input against T-skew GaR.
- Parallel-coordinates view of scenarios with especially adverse GaR.
- Later: PRIM boxes/rules that summarize input combinations associated with a pre-defined adverse outcome.

**Speaker explanation**

The empirical percentile ranges make the example traceable to the supplied data, but do not by themselves establish plausible policy scenarios. The two corner rows intentionally demonstrate why input dependence matters: a combination of marginal extremes may not occur or be economically coherent. No scenario sampling, input-to-GaR calculation, or PRIM analysis has been run for this slide.

**Sources**

- `BQR data yoy.xlsx` — `data_yoy` empirical input distributions and latest 2026Q1 row.
- `results/latest_forecast.csv` — existing reference forecast (only the first row is an evaluated result).
- `GAR/dnb_gar.py` — current one-vector `predict` interface and returned outputs.
- All sampling, histograms, scatterplots, parallel coordinates, and PRIM are proposed future work.

## Slide 8 — Research Questions and Next Steps

**Key message:** Before interpreting EMA results, the baseline’s inputs, uncertainty ranges, dependence assumptions, and evaluation criteria need to be agreed and documented.

**Slide content**

### Questions for supervisor discussion

- Is the current GDP GaR implementation a suitable starting point for studying EMA within the broader MaR thesis?
- Which inputs should be treated as uncertain, and should current `NF_YER` itself be varied alongside the other predictors?
- How should plausible ranges be justified, given that the BQR columns are prepared data and some transformations/units are not fully documented?
- Should inputs be sampled independently, or should observed economic dependencies be preserved?
- Which outcomes should be examined: T-skew GaR, direct QR 5th percentile, negative-growth probability, or additional distribution summaries?
- How should robustness and scenario discovery be evaluated and communicated?
- Which other MaR models or macroeconomic target variables should eventually be included?

### Proposed sequence

1. Confirm the baseline specification, data definitions, and reproducibility.
2. Agree on uncertain inputs, ranges, and dependence structure.
3. Design and document a sampling experiment.
4. Connect sampled vectors to the existing fitted prediction method.
5. Analyze sensitivity and, if justified, apply PRIM for scenario discovery.
6. Extend to additional MaR models or target variables only after the initial experiment is understood.

**Figure / visual**

- Roadmap with four stages: `Baseline confirmed → Sampling design → EMA evaluation → Sensitivity / PRIM and broader MaR`.
- Mark steps 2–4 as planned rather than completed.

**Speaker explanation**

The principal unresolved issue is not how to generate a large number of scenarios; it is whether those scenarios represent defensible uncertainties and relationships. Resolving the BQR transformations and setting evaluation criteria before sampling helps maintain scientific traceability and avoids presenting illustrative results as empirical evidence.

**Sources**

- `README.md` — current implementation status, data caveats, and future development.
- `METHODOLOGY.md` — model assumptions and validation limitations.
- `GAR/dnb_gar.py` — current fitted-model prediction interface.

## Existing files and results consulted

- Data and definitions: `BQR data yoy.xlsx` (`data_yoy`) and `BMPE_vintages_QLY (1).xlsx` (`List_variables`).
- Thesis entry point and adapter: `run_dnb_gar.py`, `GAR/dnb_gar.py`.
- IMF methodology implementations: `GAR/quantfit/quantilereg.py`, `GAR/tsfit/tskewfit.py`, `GAR/tsfit/tskew.py`.
- Existing fitted outputs: `results/quantile_regression_coefficients.csv`, `results/quantile_regression_diagnostics.csv`, `results/latest_forecast.csv`, `results/in_sample_conditional_quantiles.csv`.
- Existing figures: `results/latest_conditional_distribution.png`, `results/historical_growth_at_risk.png`.
- Context and caveats: `README.md`, `METHODOLOGY.md`, `THESIS_README.md`.

## Information to resolve before finalizing the slides

- Confirm the precise preprocessing and units represented by the BQR `DR_PYR` series. The BMPE list defines `PYR` as real gross personal disposable income, but that does not fully specify how the prepared `DR_PYR` column was transformed.
- Confirm definitions and units for `DF_LTN_NFCN` and the other prepared columns before interpreting coefficient magnitudes or describing scenario values as policy levers.
- Decide and document how to define plausible scenario ranges and whether predictor dependencies should be maintained.
- Keep the latest conditional forecast separate from the in-sample fitted history; an expanding-window/out-of-sample evaluation is not present in the existing artifacts.
- Check that the coefficient file and forecast file correspond to the same current specification when regenerating results for a later presentation.

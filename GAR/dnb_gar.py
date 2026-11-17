"""DNB data adapter and standalone interface for the IMF Growth-at-Risk code.

The IMF quantile-regression and T-skew fitting implementations are reused
without changing their estimation algorithms. Data loading, horizon alignment,
forecast-origin filtering, and output formatting are thesis-specific adapters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from config import (
    DATA_SHEET,
    DEPENDENT_VARIABLES,
    EMA_INPUT_LOWER_PERCENTILE,
    EMA_INPUT_UPPER_PERCENTILE,
    FORECAST_HORIZON,
    MODEL_SEQUENCE,
    MODEL_SPECS,
    OUTCOME_SPECS,
    QR_CONFIDENCE_ALPHA,
    QR_MAX_ITER,
    QR_PARAMETER_TOLERANCE,
    QUANTILE_LEVELS,
    TSKEW_REQUIRED_QUANTILES,
)
from GAR.quantfit.quantilereg import QuantileReg
from GAR.tsfit.tskew import tskew_cdf, tskew_pdf, tskew_ppf
from GAR.tsfit.tskewfit import tskew_fit


DEFAULT_PREDICTORS = MODEL_SPECS[MODEL_SEQUENCE[0]]["predictors"]
DEFAULT_DEPENDENT_VARIABLE = DEPENDENT_VARIABLES["growth"]
QUANTILES = QUANTILE_LEVELS
TARGET_COLUMN = "future_NF_YER"
TARGET_DATE_COLUMN = "target_date"
DEFAULT_HORIZON = FORECAST_HORIZON


def target_column_for(dependent_variable: str) -> str:
    """Return the aligned future-outcome column for a dependent variable."""
    return f"future_{dependent_variable}"

_TSKEW_FIT_PARAMETERS: dict[str, dict[str, Any]] = {
    "mode": {"constraint": "Free", "value": None},
    "dof": {"constraint": "Free", "value": None},
    "var_low": {"constraint": "Free", "value": None},
    "var_high": {"constraint": "Free", "value": None},
    "skew_low": {"constraint": "Free", "value": None},
    "skew_high": {"constraint": "Free", "value": None},
}


def get_model_spec(model_name: str) -> dict[str, Any]:
    """Return the thesis narrative model specification for an EMA run."""
    if model_name not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model specification {model_name!r}. "
            f"Available models: {', '.join(MODEL_SEQUENCE)}."
        )
    return MODEL_SPECS[model_name].copy()


def fit_model_spec(
    model_name: str,
    data: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
    horizon: int = DEFAULT_HORIZON,
    dependent_variable: str = DEFAULT_DEPENDENT_VARIABLE,
) -> DNBGrowthAtRisk:
    """Fit a narrative model for one configured dependent variable."""
    spec = get_model_spec(model_name)
    prepared = prepare_horizon_data(
        data,
        horizon=horizon,
        dependent_variable=dependent_variable,
        predictors=spec["predictors"],
    )
    cutoff = data.index.max() if as_of is None else as_of
    model = DNBGrowthAtRisk(
        horizon=horizon,
        predictors=spec["predictors"],
        dependent_variable=dependent_variable,
    )
    return model.fit(prepared, as_of=cutoff)


def provisional_input_bounds(
    data: pd.DataFrame,
    predictors: Sequence[str],
    lower_percentile: float = EMA_INPUT_LOWER_PERCENTILE,
    upper_percentile: float = EMA_INPUT_UPPER_PERCENTILE,
) -> dict[str, tuple[float, float]]:
    """Create provisional empirical bounds for EMA input uncertainty."""
    if not 0 <= lower_percentile < upper_percentile <= 1:
        raise ValueError(
            "EMA input percentiles must satisfy "
            "0 <= lower < upper <= 1."
        )
    bounds: dict[str, tuple[float, float]] = {}
    for name in predictors:
        series = pd.to_numeric(data[name], errors="raise")
        lower = float(series.quantile(lower_percentile))
        upper = float(series.quantile(upper_percentile))
        bounds[name] = (lower, upper)
    return bounds


def load_dnb_data(
    workbook_path: str | Path,
    sheet_name: str = DATA_SHEET,
) -> pd.DataFrame:
    """Load and validate the quarterly DNB BQR data sheet.

    Parameters
    ----------
    workbook_path : str or pathlib.Path
        Path to ``BQR data yoy.xlsx``.
    sheet_name : str, default="data_yoy"
        Worksheet containing the prepared model data.

    Returns
    -------
    pandas.DataFrame
        Numeric series indexed by strictly increasing quarter-end dates.

    Raises
    ------
    ValueError
        If the date column is invalid, dates are duplicated, or quarters are
        missing.
    """
    data = pd.read_excel(workbook_path, sheet_name=sheet_name)
    if data.empty or len(data.columns) < 2:
        raise ValueError(f"Worksheet {sheet_name!r} contains no usable data.")

    date_column = data.columns[0]
    dates = pd.to_datetime(data.pop(date_column), errors="raise")
    data.index = pd.DatetimeIndex(dates, name="date")

    if data.index.has_duplicates:
        raise ValueError("The DNB data contain duplicate dates.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("The DNB data dates must be in increasing order.")
    if not data.index.equals(data.index.to_period("Q").to_timestamp("Q")):
        raise ValueError("All DNB data dates must be quarter-end dates.")
    if len(data.index) > 1:
        quarter_steps = np.diff(data.index.to_period("Q").asi8)
        if not np.all(quarter_steps == 1):
            raise ValueError("The DNB data must contain consecutive quarters.")

    required_columns = set(DEPENDENT_VARIABLES.values())
    for spec in MODEL_SPECS.values():
        required_columns.update(spec["predictors"])
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(
            "The DNB data are missing required columns: "
            + ", ".join(missing_columns)
        )

    return data.apply(pd.to_numeric, errors="raise")


def prepare_horizon_data(
    data: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    dependent_variable: str = DEFAULT_DEPENDENT_VARIABLE,
    predictors: Sequence[str] = DEFAULT_PREDICTORS,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Align a future dependent-variable value with origin-date predictors.

    The outcome at row ``t`` is the selected dependent variable ``horizon``
    quarterly rows later. Predictors remain dated at their observation/
    forecast origin.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw quarterly data indexed by quarter-end date.
    horizon : int, default=4
        Number of quarters between predictor date and outcome date.
    dependent_variable : str, default="NF_YER"
        Series to forecast, such as GDP growth or inflation.
    predictors : sequence of str, default=the thesis predictor set
        Predictor columns observed at the forecast origin.

    Returns
    -------
    pandas.DataFrame
        Complete model rows indexed by forecast-origin date, with future
        outcomes in a ``future_<dependent_variable>`` column and their dates
        in ``target_date``.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer number of quarters.")
    required = [dependent_variable, *predictors]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Missing model columns: " + ", ".join(missing))
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Quarterly data must be indexed by a DatetimeIndex.")
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Quarterly dates must be unique and increasing.")

    outcome_column = (
        target_column_for(dependent_variable)
        if target_column is None
        else target_column
    )
    prepared = data.loc[:, list(dict.fromkeys(required))].copy()
    prepared[outcome_column] = data[dependent_variable].shift(-horizon)
    prepared[TARGET_DATE_COLUMN] = pd.Series(
        data.index, index=data.index
    ).shift(-horizon)
    prepared = prepared.dropna(
        subset=[*predictors, outcome_column, TARGET_DATE_COLUMN]
    )
    prepared[TARGET_DATE_COLUMN] = pd.to_datetime(
        prepared[TARGET_DATE_COLUMN]
    )
    return prepared


class DNBGrowthAtRisk:
    """Fit an IMF-based DNB conditional-risk model.

    Parameters
    ----------
    horizon : int, default=4
        Forecast horizon in quarters.
    predictors : sequence of str, default=the thesis predictor set
        Input variables used by the quantile regressions.
    quantiles : sequence of float, default=(0.05, 0.10, ..., 0.95)
        Conditional quantiles fitted by the IMF quantile-regression code.
    dependent_variable : str, default="NF_YER"
        Outcome series to forecast, including GDP growth or inflation.

    Notes
    -----
    ``fit`` requires a cutoff date and excludes training rows whose target
    observations were not available by that date. ``predict`` accepts a single
    mapping of predictor values, making the fitted model callable without
    Excel or plotting.
    """

    def __init__(
        self,
        horizon: int = DEFAULT_HORIZON,
        predictors: Sequence[str] = DEFAULT_PREDICTORS,
        quantiles: Sequence[float] = QUANTILES,
        dependent_variable: str = DEFAULT_DEPENDENT_VARIABLE,
    ) -> None:
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon must be a positive integer number of quarters.")
        if len(set(predictors)) != len(predictors) or not predictors:
            raise ValueError("predictors must be a non-empty sequence of unique names.")
        if not set(TSKEW_REQUIRED_QUANTILES).issubset(quantiles):
            raise ValueError(
                "The IMF T-skew fit requires 5th, 25th, 50th, and 75th quantiles."
            )
        if any(not 0 < quantile < 1 for quantile in quantiles):
            raise ValueError("All quantiles must be strictly between zero and one.")

        self.horizon = horizon
        self.predictors = tuple(predictors)
        self.quantiles = tuple(sorted(set(float(q) for q in quantiles)))
        self.dependent_variable = dependent_variable
        self.target_column = target_column_for(dependent_variable)
        self.regression: QuantileReg | None = None
        self.training_data: pd.DataFrame | None = None
        self.as_of: pd.Timestamp | None = None
        self.fit_diagnostics: pd.DataFrame | None = None

    def fit(
        self,
        prepared_data: pd.DataFrame,
        as_of: str | pd.Timestamp,
    ) -> DNBGrowthAtRisk:
        """Estimate all conditional quantile regressions available by a cutoff.

        Parameters
        ----------
        prepared_data : pandas.DataFrame
            Output of :func:`prepare_horizon_data`.
        as_of : str or pandas.Timestamp
            Information cutoff; training rows whose ``target_date`` is later
            than this date are excluded.

        Returns
        -------
        DNBGrowthAtRisk
            This fitted model instance.
        """
        required_columns = [
            *self.predictors,
            self.target_column,
            TARGET_DATE_COLUMN,
        ]
        missing = [
            column for column in required_columns if column not in prepared_data
        ]
        if missing:
            raise ValueError(
                "Prepared data are missing required columns: "
                + ", ".join(missing)
            )
        cutoff = pd.Timestamp(as_of)
        target_dates = pd.to_datetime(prepared_data[TARGET_DATE_COLUMN])
        training = prepared_data.loc[target_dates <= cutoff].copy()
        training = training.dropna(subset=required_columns)
        if len(training) <= len(self.predictors) + 1:
            raise ValueError(
                "Not enough complete training rows are available at the cutoff."
            )
        values = training.loc[
            :, [*self.predictors, self.target_column]
        ].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Training data contain non-finite numeric values.")
        if any(training[column].nunique() < 2 for column in self.predictors):
            raise ValueError("Each predictor must vary in the training data.")

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=IterationLimitWarning)
            try:
                regression = QuantileReg(
                    self.target_column,
                    indvars=list(self.predictors),
                    quantile_list=list(self.quantiles),
                    data=training,
                    scaling=False,
                    alpha=QR_CONFIDENCE_ALPHA,
                    max_iter=QR_MAX_ITER,
                    parameter_tolerance=QR_PARAMETER_TOLERANCE,
                )
            except IterationLimitWarning as error:
                raise RuntimeError(
                    "At least one IMF quantile regression did not converge."
                ) from error

        diagnostics = []
        for quantile, result in regression.qfit_dict.items():
            if not np.isfinite(result.params.to_numpy()).all():
                raise RuntimeError(
                    f"The {quantile:.0%} quantile regression returned "
                    "non-finite coefficients."
                )
            diagnostics.append(
                {
                    "quantile": quantile,
                    "observations": int(result.nobs),
                    "iterations": getattr(result, "iterations", np.nan),
                }
            )

        self.regression = regression
        self.training_data = training
        self.as_of = cutoff
        self.fit_diagnostics = pd.DataFrame(diagnostics)
        return self

    def predict(self, predictors: Mapping[str, float]) -> dict[str, Any]:
        """Return conditional quantiles and an IMF T-skew distribution.

        Parameters
        ----------
        predictors : mapping of str to float
            One value per predictor, in the same units/transformation as the
            corresponding BQR worksheet column.

        Returns
        -------
        dict
            Quantile-regression estimates, T-skew parameters, T-skew quantiles,
            and the T-skew cumulative probability at zero growth.
        """
        if self.regression is None:
            raise RuntimeError("Call fit before predict.")
        missing = [name for name in self.predictors if name not in predictors]
        if missing:
            raise ValueError("Missing predictor values: " + ", ".join(missing))
        values = {name: float(predictors[name]) for name in self.predictors}
        if not np.isfinite(list(values.values())).all():
            raise ValueError("Predictor values must be finite numbers.")

        scenario = pd.DataFrame([values], columns=self.predictors)
        quantile_values = {
            quantile: float(result.predict(scenario).iloc[0])
            for quantile, result in self.regression.qfit_dict.items()
        }
        distribution = tskew_fit(
            quantile_values.copy(),
            _TSKEW_FIT_PARAMETERS,
        )
        parameters = {
            name: float(distribution[name])
            for name in ("loc", "scale", "skew")
        }
        parameters["df"] = int(distribution["df"])
        if (
            not np.isfinite(list(parameters.values())).all()
            or parameters["scale"] <= 0
            or parameters["skew"] <= 0
        ):
            raise RuntimeError("The IMF T-skew fit returned invalid parameters.")

        distribution_quantiles = {
            quantile: float(tskew_ppf(quantile, **parameters))
            for quantile in self.quantiles
        }
        sorted_values = [
            distribution_quantiles[quantile] for quantile in self.quantiles
        ]
        if not np.isfinite(sorted_values).all() or not np.all(
            np.diff(sorted_values) > 0
        ):
            raise RuntimeError(
                "The fitted IMF T-skew distribution is not strictly increasing."
            )

        outcome_key = next(
            (
                key
                for key, spec in OUTCOME_SPECS.items()
                if spec["dependent_variable"] == self.dependent_variable
            ),
            None,
        )
        primary_quantile = (
            float(OUTCOME_SPECS[outcome_key]["primary_quantile"])
            if outcome_key is not None
            else 0.05
        )
        prediction = {
            "quantile_regression": quantile_values,
            "distribution_parameters": parameters,
            "distribution_quantiles": distribution_quantiles,
            "probability_below_zero": float(tskew_cdf(0.0, **parameters)),
            "primary_tail_quantile": primary_quantile,
            "primary_quantile_regression": quantile_values[primary_quantile],
            "primary_distribution_quantile": distribution_quantiles[
                primary_quantile
            ],
        }
        if self.dependent_variable == DEPENDENT_VARIABLES["growth"]:
            prediction["probability_growth_below_zero"] = prediction[
                "probability_below_zero"
            ]
        return prediction


def write_model_results(
    model: DNBGrowthAtRisk,
    raw_data: pd.DataFrame,
    prepared_data: pd.DataFrame,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Write model tables and figures without modifying either input workbook.

    In-sample fitted values use the full estimation sample and are explicitly
    labelled as such; they are not presented as historical real-time forecasts.
    The separate latest forecast uses only outcomes available by its origin.

    Parameters
    ----------
    model : DNBGrowthAtRisk
        Fitted model.
    raw_data : pandas.DataFrame
        Original date-indexed BQR data.
    prepared_data : pandas.DataFrame
        Horizon-aligned model rows.
    output_directory : str or pathlib.Path
        Directory where CSV and PNG outputs are written.

    Returns
    -------
    dict of str to pathlib.Path
        Paths to the generated results.
    """
    if model.regression is None or model.training_data is None:
        raise RuntimeError("Call fit before writing results.")

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    coefficients = model.regression.coeff.copy()
    coefficients.index.name = "term"
    coefficients = coefficients.reset_index()
    coefficients.to_csv(
        output_dir / "quantile_regression_coefficients.csv", index=False
    )
    if model.fit_diagnostics is not None:
        model.fit_diagnostics.to_csv(
            output_dir / "quantile_regression_diagnostics.csv", index=False
        )

    outcome_key = next(
        (
            key
            for key, spec in OUTCOME_SPECS.items()
            if spec["dependent_variable"] == model.dependent_variable
        ),
        "growth",
    )
    outcome_spec = OUTCOME_SPECS[outcome_key]
    primary_quantile = float(outcome_spec["primary_quantile"])
    dependent_variable = model.dependent_variable
    target_column = model.target_column
    outcome_label = str(outcome_spec["label"])
    risk_name = str(outcome_spec["risk_name"])
    outcome_unit = str(outcome_spec["unit"])

    in_sample_rows = []
    crossing_counts = {quantile: 0 for quantile in model.quantiles[:-1]}
    rows_with_crossing = 0
    for origin_date, row in prepared_data.iterrows():
        prediction = model.predict(
            {name: float(row[name]) for name in model.predictors}
        )
        qr_values = [
            prediction["quantile_regression"][quantile]
            for quantile in model.quantiles
        ]
        crossed = np.diff(qr_values) <= 0
        rows_with_crossing += int(crossed.any())
        for index, has_crossing in enumerate(crossed):
            crossing_counts[model.quantiles[index]] += int(has_crossing)
        output_row: dict[str, Any] = {
            "forecast_origin": origin_date,
            "target_date": row[TARGET_DATE_COLUMN],
            "horizon_quarters": model.horizon,
            f"realized_future_{dependent_variable}": float(row[target_column]),
            "fit_status": "full_sample_in_sample_fit",
        }
        for quantile, value in prediction["quantile_regression"].items():
            output_row[f"qr_q{int(quantile * 100):02d}"] = value
        for quantile, value in prediction["distribution_quantiles"].items():
            output_row[f"tskew_q{int(quantile * 100):02d}"] = value
        output_row.update(
            {
                f"tskew_{name}": value
                for name, value in prediction["distribution_parameters"].items()
            }
        )
        output_row["probability_below_zero"] = prediction[
            "probability_below_zero"
        ]
        if dependent_variable == DEPENDENT_VARIABLES["growth"]:
            output_row["probability_growth_below_zero"] = prediction[
                "probability_below_zero"
            ]
        in_sample_rows.append(output_row)

    in_sample = pd.DataFrame(in_sample_rows)
    in_sample.to_csv(output_dir / "in_sample_conditional_quantiles.csv", index=False)
    crossing_diagnostics = pd.DataFrame(
        [
            {
                "quantile_pair": (
                    f"q{int(quantile * 100):02d}_q"
                    f"{int(next_quantile * 100):02d}"
                ),
                "crossed_rows": crossing_counts[quantile],
                "rows_evaluated": len(in_sample),
            }
            for quantile, next_quantile in zip(
                model.quantiles[:-1], model.quantiles[1:]
            )
        ]
    )
    crossing_diagnostics["rows_with_any_crossing"] = rows_with_crossing
    crossing_diagnostics.to_csv(
        output_dir / "quantile_crossing_diagnostics.csv", index=False
    )

    latest_origin = raw_data.index[-1]
    latest_predictors = raw_data.loc[latest_origin, list(model.predictors)]
    latest_prediction = model.predict(
        {name: float(latest_predictors[name]) for name in model.predictors}
    )
    target_date = latest_origin + pd.offsets.QuarterEnd(model.horizon)
    latest_row: dict[str, Any] = {
        "forecast_origin": latest_origin,
        "target_date": target_date,
        "horizon_quarters": model.horizon,
        "information_cutoff": model.as_of,
        "primary_risk_name": risk_name,
        "primary_tail_quantile": primary_quantile,
        "primary_qr_tail": latest_prediction["primary_quantile_regression"],
        "primary_tskew_tail": latest_prediction[
            "primary_distribution_quantile"
        ],
        "probability_below_zero": latest_prediction["probability_below_zero"],
    }
    if dependent_variable == DEPENDENT_VARIABLES["growth"]:
        latest_row.update(
            {
                "qr_q05": latest_prediction["quantile_regression"][0.05],
                "tskew_q05_GaR": latest_prediction[
                    "distribution_quantiles"
                ][0.05],
                "probability_growth_below_zero": latest_prediction[
                    "probability_below_zero"
                ],
            }
        )
    elif dependent_variable == DEPENDENT_VARIABLES["inflation"]:
        latest_row["tskew_q95_Inflation_at_Risk"] = latest_prediction[
            "distribution_quantiles"
        ][0.95]
    for quantile, value in latest_prediction["quantile_regression"].items():
        latest_row[f"qr_q{int(quantile * 100):02d}"] = value
    for quantile, value in latest_prediction["distribution_quantiles"].items():
        latest_row[f"tskew_q{int(quantile * 100):02d}"] = value
    for name, value in latest_prediction["distribution_parameters"].items():
        latest_row[f"tskew_{name}"] = value
    pd.DataFrame([latest_row]).to_csv(
        output_dir / "latest_forecast.csv", index=False
    )
    training = model.training_data
    if training is None:
        raise RuntimeError("The fitted model has no retained training data.")
    target_dates = pd.to_datetime(training[TARGET_DATE_COLUMN])
    model_summary = {
        "dependent_variable": dependent_variable,
        "outcome": outcome_label,
        "risk_name": risk_name,
        "primary_tail_quantile": primary_quantile,
        "predictors": list(model.predictors),
        "forecast_horizon_quarters": model.horizon,
        "estimation_observations": len(training),
        "forecast_origin_start": training.index.min().strftime("%Y-%m-%d"),
        "forecast_origin_end": training.index.max().strftime("%Y-%m-%d"),
        "target_date_start": target_dates.min().strftime("%Y-%m-%d"),
        "target_date_end": target_dates.max().strftime("%Y-%m-%d"),
        "quantile_levels": list(model.quantiles),
        "direct_qr_rows_with_any_crossing": rows_with_crossing,
        "direct_qr_rows_evaluated": len(in_sample),
        "inflation_transformation_note": (
            "NF_HIC is used as supplied. Values appear consistent with an "
            "inflation rate rather than a positive index level, but the "
            "available workbook does not document its exact transformation."
            if dependent_variable == DEPENDENT_VARIABLES["inflation"]
            else None
        ),
    }
    with (output_dir / "model_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(model_summary, file, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary_suffix = f"q{int(primary_quantile * 100):02d}"
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        in_sample["forecast_origin"],
        in_sample[f"tskew_{primary_suffix}"],
        color="firebrick",
        label=f"Fitted T-skew {primary_suffix} ({risk_name})",
    )
    axis.plot(
        in_sample["forecast_origin"],
        in_sample[f"realized_future_{dependent_variable}"],
        color="black",
        alpha=0.55,
        label=(
            f"Realized {dependent_variable} at origin + "
            f"{model.horizon} quarters"
        ),
    )
    axis.set_title(
        f"In-sample fitted {risk_name}: {outcome_label}, "
        f"{model.horizon} quarters ahead"
    )
    axis.set_xlabel("Forecast origin (quarter-end)")
    axis.set_ylabel(outcome_unit)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    history_filename = (
        "historical_growth_at_risk.png"
        if outcome_key == "growth"
        else "historical_inflation_at_risk.png"
    )
    history_path = output_dir / history_filename
    fig.savefig(history_path, dpi=160)
    plt.close(fig)

    parameters = latest_prediction["distribution_parameters"]
    x_values = np.linspace(
        tskew_ppf(0.005, **parameters),
        tskew_ppf(0.995, **parameters),
        500,
    )
    density = [tskew_pdf(float(value), **parameters) for value in x_values]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(x_values, density, color="navy")
    tail_value = latest_prediction["distribution_quantiles"][primary_quantile]
    if primary_quantile < 0.5:
        tail_mask = x_values <= tail_value
        tail_label = f"Lower {primary_quantile:.0%} tail"
        tail_color = "firebrick"
    else:
        tail_mask = x_values >= tail_value
        tail_label = f"Upper {(1 - primary_quantile):.0%} tail"
        tail_color = "darkorange"
    axis.fill_between(
        x_values,
        0,
        density,
        where=tail_mask,
        color=tail_color,
        alpha=0.35,
        label=tail_label,
    )
    axis.axvline(
        tail_value,
        color=tail_color,
        linestyle="--",
        label=f"{risk_name} ({primary_suffix})",
    )
    axis.set_title(
        f"Latest conditional T-skew: {outcome_label}, "
        f"{latest_origin:%Y-%m-%d} origin, {model.horizon}-quarter horizon"
    )
    axis.set_xlabel(outcome_unit)
    axis.set_ylabel("Probability density")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    distribution_path = output_dir / "latest_conditional_distribution.png"
    fig.savefig(distribution_path, dpi=160)
    plt.close(fig)

    return {
        "coefficients": output_dir / "quantile_regression_coefficients.csv",
        "diagnostics": output_dir / "quantile_regression_diagnostics.csv",
        "crossing_diagnostics": output_dir / "quantile_crossing_diagnostics.csv",
        "model_summary": output_dir / "model_summary.json",
        "in_sample": output_dir / "in_sample_conditional_quantiles.csv",
        "latest_forecast": output_dir / "latest_forecast.csv",
        "historical_figure": history_path,
        "distribution_figure": distribution_path,
    }

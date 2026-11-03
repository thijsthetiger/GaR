"""DNB data adapter and standalone interface for the IMF Growth-at-Risk code.

The IMF quantile-regression and T-skew fitting implementations are reused
without changing their estimation algorithms. Data loading, horizon alignment,
forecast-origin filtering, and output formatting are thesis-specific adapters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from GAR.quantfit.quantilereg import QuantileReg
from GAR.tsfit.tskew import tskew_cdf, tskew_pdf, tskew_ppf
from GAR.tsfit.tskewfit import tskew_fit


DEPENDENT_VARIABLE = "NF_YER"
PREDICTORS = ("NF_YER", "NF_URX", "DF_LTN_NFCN", "DR_PYR")
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
TARGET_COLUMN = "future_NF_YER"
TARGET_DATE_COLUMN = "target_date"
DATA_SHEET = "data_yoy"
DEFAULT_HORIZON = 4

_TSKEW_FIT_PARAMETERS: dict[str, dict[str, Any]] = {
    "mode": {"constraint": "Free", "value": None},
    "dof": {"constraint": "Free", "value": None},
    "var_low": {"constraint": "Free", "value": None},
    "var_high": {"constraint": "Free", "value": None},
    "skew_low": {"constraint": "Free", "value": None},
    "skew_high": {"constraint": "Free", "value": None},
}


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

    missing_columns = [
        column
        for column in (DEPENDENT_VARIABLE, *PREDICTORS)
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "The DNB data are missing required columns: "
            + ", ".join(missing_columns)
        )

    return data.apply(pd.to_numeric, errors="raise")


def prepare_horizon_data(
    data: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    dependent_variable: str = DEPENDENT_VARIABLE,
    predictors: Sequence[str] = PREDICTORS,
) -> pd.DataFrame:
    """Align future GDP growth with predictor values at the forecast origin.

    The outcome at row ``t`` is the dependent variable ``horizon`` quarterly
    rows later. Predictors remain dated at their observation/forecast origin.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw quarterly data indexed by quarter-end date.
    horizon : int, default=4
        Number of quarters between predictor date and outcome date.
    dependent_variable : str, default="NF_YER"
        Series to forecast.
    predictors : sequence of str, default=the thesis predictor set
        Predictor columns observed at the forecast origin.

    Returns
    -------
    pandas.DataFrame
        Complete model rows indexed by forecast-origin date, with future
        outcomes in ``future_NF_YER`` and their dates in ``target_date``.
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

    prepared = data.loc[:, list(dict.fromkeys(required))].copy()
    prepared[TARGET_COLUMN] = data[dependent_variable].shift(-horizon)
    prepared[TARGET_DATE_COLUMN] = pd.Series(
        data.index, index=data.index
    ).shift(-horizon)
    prepared = prepared.dropna(
        subset=[*predictors, TARGET_COLUMN, TARGET_DATE_COLUMN]
    )
    prepared[TARGET_DATE_COLUMN] = pd.to_datetime(
        prepared[TARGET_DATE_COLUMN]
    )
    return prepared


class DNBGrowthAtRisk:
    """Fit the IMF-based DNB GaR model and predict from macroeconomic inputs.

    Parameters
    ----------
    horizon : int, default=4
        Forecast horizon in quarters.
    predictors : sequence of str, default=the thesis predictor set
        Input variables used by the quantile regressions.
    quantiles : sequence of float, default=(0.05, 0.10, ..., 0.95)
        Conditional quantiles fitted by the IMF quantile-regression code.

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
        predictors: Sequence[str] = PREDICTORS,
        quantiles: Sequence[float] = QUANTILES,
    ) -> None:
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon must be a positive integer number of quarters.")
        if len(set(predictors)) != len(predictors) or not predictors:
            raise ValueError("predictors must be a non-empty sequence of unique names.")
        if not set((0.05, 0.25, 0.50, 0.75)).issubset(quantiles):
            raise ValueError(
                "The IMF T-skew fit requires 5th, 25th, 50th, and 75th quantiles."
            )
        if any(not 0 < quantile < 1 for quantile in quantiles):
            raise ValueError("All quantiles must be strictly between zero and one.")

        self.horizon = horizon
        self.predictors = tuple(predictors)
        self.quantiles = tuple(sorted(set(float(q) for q in quantiles)))
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
            TARGET_COLUMN,
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
        values = training.loc[:, [*self.predictors, TARGET_COLUMN]].to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError("Training data contain non-finite numeric values.")
        if any(training[column].nunique() < 2 for column in self.predictors):
            raise ValueError("Each predictor must vary in the training data.")

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=IterationLimitWarning)
            try:
                regression = QuantileReg(
                    TARGET_COLUMN,
                    indvars=list(self.predictors),
                    quantile_list=list(self.quantiles),
                    data=training,
                    scaling=False,
                    alpha=0.1,
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

        return {
            "quantile_regression": quantile_values,
            "distribution_parameters": parameters,
            "distribution_quantiles": distribution_quantiles,
            "probability_growth_below_zero": float(tskew_cdf(0.0, **parameters)),
        }


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

    in_sample_rows = []
    for origin_date, row in prepared_data.iterrows():
        prediction = model.predict(
            {name: float(row[name]) for name in model.predictors}
        )
        output_row: dict[str, Any] = {
            "forecast_origin": origin_date,
            "target_date": row[TARGET_DATE_COLUMN],
            "horizon_quarters": model.horizon,
            "realized_future_NF_YER": float(row[TARGET_COLUMN]),
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
        output_row["probability_growth_below_zero"] = prediction[
            "probability_growth_below_zero"
        ]
        in_sample_rows.append(output_row)

    in_sample = pd.DataFrame(in_sample_rows)
    in_sample.to_csv(output_dir / "in_sample_conditional_quantiles.csv", index=False)

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
        "qr_q05": latest_prediction["quantile_regression"][0.05],
        "tskew_q05_GaR": latest_prediction["distribution_quantiles"][0.05],
        "probability_growth_below_zero": latest_prediction[
            "probability_growth_below_zero"
        ],
    }
    for quantile, value in latest_prediction["quantile_regression"].items():
        latest_row[f"qr_q{int(quantile * 100):02d}"] = value
    for quantile, value in latest_prediction["distribution_quantiles"].items():
        latest_row[f"tskew_q{int(quantile * 100):02d}"] = value
    for name, value in latest_prediction["distribution_parameters"].items():
        latest_row[f"tskew_{name}"] = value
    pd.DataFrame([latest_row]).to_csv(
        output_dir / "latest_forecast.csv", index=False
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        in_sample["forecast_origin"],
        in_sample["tskew_q05"],
        color="firebrick",
        label="T-skew fitted 5th percentile (GaR)",
    )
    axis.plot(
        in_sample["forecast_origin"],
        in_sample["realized_future_NF_YER"],
        color="black",
        alpha=0.55,
        label=f"Realized NF_YER at origin + {model.horizon} quarters",
    )
    axis.set_title(
        f"In-sample fitted Growth-at-Risk, {model.horizon} quarters ahead"
    )
    axis.set_xlabel("Forecast origin (quarter-end)")
    axis.set_ylabel("Year-on-year real GDP growth (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    history_path = output_dir / "historical_growth_at_risk.png"
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
    axis.fill_between(
        x_values,
        0,
        density,
        where=x_values <= latest_prediction["distribution_quantiles"][0.05],
        color="firebrick",
        alpha=0.35,
        label="Lower 5% tail",
    )
    axis.axvline(
        latest_prediction["distribution_quantiles"][0.05],
        color="firebrick",
        linestyle="--",
        label="T-skew GaR (5th percentile)",
    )
    axis.set_title(
        f"Latest conditional T-skew distribution: "
        f"{latest_origin:%Y-%m-%d} origin, {model.horizon}-quarter horizon"
    )
    axis.set_xlabel("Year-on-year real GDP growth (%)")
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
        "in_sample": output_dir / "in_sample_conditional_quantiles.csv",
        "latest_forecast": output_dir / "latest_forecast.csv",
        "historical_figure": history_path,
        "distribution_figure": distribution_path,
    }

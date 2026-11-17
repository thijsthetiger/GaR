"""Run EMA input-sampling experiments for the thesis GaR narrative models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DATA_FILE,
    DATA_SHEET,
    EMA_RANDOM_SEED,
    EMA_RESULTS_DIR,
    EMA_SCENARIO_COUNT,
    EMA_SAMPLING_METHOD,
    EMA_INPUT_LOWER_PERCENTILE,
    EMA_INPUT_UPPER_PERCENTILE,
    MODEL_SEQUENCE,
    MODEL_SPECS,
    MODEL_OUTPUT_NAMES,
    OUTCOME_SEQUENCE,
    OUTCOME_SPECS,
    QUANTILE_LEVELS,
)
from ema_workbench import (  # noqa: E402
    Model,
    RealParameter,
    ScalarOutcome,
    SequentialEvaluator,
    perform_experiments,
    util,
)
from ema_workbench.em_framework.samplers import LHSSampler  # noqa: E402

from GAR.dnb_gar import (  # noqa: E402
    fit_model_spec,
    load_dnb_data,
    provisional_input_bounds,
)
from GAR.tsfit.tskew import tskew_pdf  # noqa: E402


N_SCENARIOS = EMA_SCENARIO_COUNT
RANDOM_SEED = EMA_RANDOM_SEED
OUTPUT_DIRECTORY = EMA_RESULTS_DIR


def _jsonify(value: Any) -> Any:
    """Convert NumPy scalars to JSON-serialisable Python values."""
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _percentile_column(prefix: str, percentile: float) -> str:
    """Return a stable CSV column name for a configured percentile."""
    return f"{prefix}_p{int(round(percentile * 100)):02d}"


def _scenario_identifier(
    narrative_id: str,
    inputs: dict[str, float],
    predictors: tuple[str, ...],
) -> str:
    """Create a stable ID from a narrative and exact sampled input values."""
    payload = json.dumps(
        [
            narrative_id,
            [(name, float(inputs[name]).hex()) for name in predictors],
        ],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prediction_outcomes(
    outcome_name: str,
    prediction: dict[str, Any],
) -> dict[str, float]:
    """Flatten all fitted quantiles and parameters for an EMA outcome."""
    qr_values = [
        float(prediction["quantile_regression"][quantile])
        for quantile in QUANTILE_LEVELS
    ]
    crossing_pairs = np.diff(qr_values) <= 0
    output = {
        f"{outcome_name}_qr_q{int(round(quantile * 100)):02d}": float(value)
        for quantile, value in prediction["quantile_regression"].items()
    }
    output.update(
        {
            f"{outcome_name}_tskew_q{int(round(quantile * 100)):02d}": float(
                value
            )
            for quantile, value in prediction["distribution_quantiles"].items()
        }
    )
    output.update(
        {
            f"{outcome_name}_tskew_{name}": float(value)
            for name, value in prediction["distribution_parameters"].items()
        }
    )
    output[f"{outcome_name}_primary_tail_quantile"] = float(
        prediction["primary_tail_quantile"]
    )
    output[f"{outcome_name}_primary_tail_risk"] = float(
        prediction["primary_distribution_quantile"]
    )
    output[f"{outcome_name}_qr_crossing_pair_count"] = float(
        crossing_pairs.sum()
    )
    output[f"{outcome_name}_qr_any_crossing"] = float(crossing_pairs.any())
    if outcome_name == "growth":
        output["growth_probability_below_zero"] = float(
            prediction["probability_below_zero"]
        )
    return output


def _outcome_columns(outcome_name: str) -> list[str]:
    """Return all numeric outcome names expected from one EMA evaluation."""
    quantile_columns = [
        f"{outcome_name}_{kind}_q{int(round(quantile * 100)):02d}"
        for kind in ("qr", "tskew")
        for quantile in QUANTILE_LEVELS
    ]
    columns = [
        *quantile_columns,
        *[
            f"{outcome_name}_tskew_{parameter}"
            for parameter in ("loc", "scale", "skew", "df")
        ],
        f"{outcome_name}_primary_tail_quantile",
        f"{outcome_name}_primary_tail_risk",
        f"{outcome_name}_qr_crossing_pair_count",
        f"{outcome_name}_qr_any_crossing",
    ]
    if outcome_name == "growth":
        columns.append("growth_probability_below_zero")
    return columns


def _write_figures(
    scenarios: pd.DataFrame,
    reference_prediction: dict[str, Any],
    output_directory: Path,
    model_predictors: tuple[str, ...],
    outcome_name: str,
) -> None:
    """Write outcome-labelled plots for one narrative EMA experiment."""
    outcome_spec = OUTCOME_SPECS[outcome_name]
    primary_quantile = float(outcome_spec["primary_quantile"])
    primary_column = (
        f"{outcome_name}_tskew_q{int(round(primary_quantile * 100)):02d}"
    )
    primary_risk = scenarios[primary_column]
    reference_risk = float(
        reference_prediction["distribution_quantiles"][primary_quantile]
    )
    outcome_label = str(outcome_spec["label"])
    risk_name = str(outcome_spec["risk_name"])
    output_tag = outcome_name

    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.hist(primary_risk, bins=15, color="#4472C4", edgecolor="white")
    axis.axvline(
        reference_risk,
        color="#C00000",
        linestyle="--",
        linewidth=2,
        label=f"Reference {risk_name}: {reference_risk:.3f}",
    )
    axis.set(
        xlabel=(
            f"Fitted T-skew {primary_quantile:.0%} percentile "
            f"({outcome_spec['unit']})"
        ),
        ylabel="Number of sampled input scenarios",
        title=f"{risk_name} across sampled input scenarios",
    )
    axis.legend()
    fig.savefig(
        output_directory / f"figure1_{output_tag}_scenario_histogram.png",
        dpi=220,
    )
    plt.close(fig)

    columns = 2
    rows = (len(model_predictors) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, name in zip(axes.flat, model_predictors):
        axis.scatter(
            scenarios[name],
            primary_risk,
            alpha=0.8,
            color="#4472C4",
            edgecolors="white",
            linewidths=0.4,
        )
        axis.axhline(reference_risk, color="#C00000", linestyle="--", linewidth=1)
        axis.set_xlabel(f"{name} (BQR column units)")
        axis.set_ylabel(f"Fitted T-skew {risk_name} ({outcome_spec['unit']})")
        axis.set_title(f"{name} versus {risk_name}")
    for axis in axes.flat[len(model_predictors) :]:
        axis.set_visible(False)
    figure.savefig(
        output_directory / f"figure2_{output_tag}_inputs_vs_tail_risk.png",
        dpi=220,
    )
    plt.close(figure)

    adverse_index = (
        primary_risk.idxmin()
        if primary_quantile < 0.5
        else primary_risk.idxmax()
    )
    favorable_index = (
        primary_risk.idxmax()
        if primary_quantile < 0.5
        else primary_risk.idxmin()
    )
    adverse = scenarios.loc[adverse_index]
    favorable = scenarios.loc[favorable_index]
    scenario_predictions = {
        "Reference": reference_prediction,
        "Most adverse sampled": {
            "distribution_parameters": {
                name: float(adverse[f"{outcome_name}_tskew_{name}"])
                for name in ("loc", "scale", "skew")
            }
            | {"df": int(adverse[f"{outcome_name}_tskew_df"])},
            "distribution_quantiles": {
                quantile: float(
                    adverse[
                        f"{outcome_name}_tskew_q{int(round(quantile * 100)):02d}"
                    ]
                )
                for quantile in (0.05, 0.25, 0.50, 0.75, 0.95)
            },
        },
        "Most favorable sampled": {
            "distribution_parameters": {
                name: float(favorable[f"{outcome_name}_tskew_{name}"])
                for name in ("loc", "scale", "skew")
            }
            | {"df": int(favorable[f"{outcome_name}_tskew_df"])},
            "distribution_quantiles": {
                quantile: float(
                    favorable[
                        f"{outcome_name}_tskew_q{int(round(quantile * 100)):02d}"
                    ]
                )
                for quantile in (0.05, 0.25, 0.50, 0.75, 0.95)
            },
        },
    }

    grid_min = min(
        float(min(prediction["distribution_quantiles"].values()))
        for prediction in scenario_predictions.values()
    )
    grid_max = max(
        float(max(prediction["distribution_quantiles"].values()))
        for prediction in scenario_predictions.values()
    )
    padding = max((grid_max - grid_min) * 0.1, 1.0)
    x_values = np.linspace(grid_min - padding, grid_max + padding, 600)
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = ("#1F4E79", "#C00000", "#70AD47")
    for (label, prediction), color in zip(scenario_predictions.items(), colors):
        parameters = prediction["distribution_parameters"]
        density = np.array(
            [tskew_pdf(float(x), **parameters) for x in x_values]
        )
        tail_value = float(
            prediction["distribution_quantiles"][primary_quantile]
        )
        axis.plot(x_values, density, color=color, linewidth=2, label=label)
        axis.axvline(tail_value, color=color, linestyle="--", linewidth=1.5)
        axis.text(
            tail_value,
            float(density.max()),
            f" {primary_quantile:.0%}={tail_value:.2f}",
            color=color,
            rotation=90,
            va="top",
            ha="right",
        )
    axis.set(
        xlabel=str(outcome_spec["unit"]),
        ylabel="T-skew probability density",
        title=f"Conditional {outcome_label} distributions for selected inputs",
    )
    axis.legend()
    fig.savefig(
        output_directory / f"figure3_{output_tag}_conditional_distributions.png",
        dpi=220,
    )
    plt.close(fig)


def _write_outcome_results(
    scenario_results: pd.DataFrame,
    failures: list[dict[str, Any]],
    model: Any,
    reference_prediction: dict[str, Any],
    reference_inputs: dict[str, float],
    predictors: tuple[str, ...],
    outcome_name: str,
    narrative_id: str,
    output_directory: Path,
    input_bounds: dict[str, tuple[float, float]],
    input_observations: dict[str, int],
) -> pd.DataFrame:
    """Write one outcome's results from the paired narrative experiment."""
    output_directory.mkdir(parents=True, exist_ok=True)
    outcome_prefix = f"{outcome_name}_"
    output_columns = [
        column
        for column in scenario_results.columns
        if column.startswith(outcome_prefix)
    ]
    scenario_frame = scenario_results.loc[
        :, ["narrative_id", "scenario_id", *predictors, *output_columns]
    ].copy()
    failed_column = f"{outcome_name}_evaluation_failed"
    scenario_frame[failed_column] = scenario_results[failed_column].astype(int)

    if outcome_name == "growth":
        scenario_frame["QR_q05"] = scenario_frame["growth_qr_q05"]
        scenario_frame["Tskew_GaR_q05"] = scenario_frame["growth_tskew_q05"]
        scenario_frame["Tskew_Median"] = scenario_frame["growth_tskew_q50"]
        scenario_frame["Probability_Negative"] = scenario_frame[
            "growth_probability_below_zero"
        ]
    else:
        scenario_frame["QR_q95"] = scenario_frame["inflation_qr_q95"]
        scenario_frame["Tskew_IaR_q95"] = scenario_frame[
            "inflation_tskew_q95"
        ]
        scenario_frame["Tskew_Median"] = scenario_frame["inflation_tskew_q50"]
    legacy_parameter_names = {
        "loc": "Location",
        "scale": "Scale",
        "skew": "Skew",
        "df": "DF",
    }
    for parameter in ("loc", "scale", "skew", "df"):
        scenario_frame[
            f"Tskew_{legacy_parameter_names[parameter]}"
        ] = scenario_frame[
            f"{outcome_name}_tskew_{parameter}"
        ]
    scenario_frame["Evaluation_Failed"] = scenario_frame[failed_column]
    scenario_frame.to_csv(output_directory / "scenario_results.csv", index=False)

    failure_rows = [
        failure for failure in failures if failure["outcome"] == outcome_name
    ]
    pd.DataFrame(
        failure_rows,
        columns=[
            "narrative_id",
            "scenario_id",
            "outcome",
            *predictors,
            "error_type",
            "error_message",
        ],
    ).to_csv(output_directory / "scenario_failures.csv", index=False)
    if model.fit_diagnostics is not None:
        model.fit_diagnostics.to_csv(
            output_directory / "quantile_regression_diagnostics.csv",
            index=False,
        )

    bounds_rows = [
        {
            "variable": name,
            _percentile_column("lower", EMA_INPUT_LOWER_PERCENTILE): bounds[0],
            _percentile_column("upper", EMA_INPUT_UPPER_PERCENTILE): bounds[1],
            "bound_sample": "full available predictor history",
            "observations": input_observations[name],
            "unit_note": (
                "Prepared BQR column units; exact transformations/units "
                "are not fully documented."
            ),
        }
        for name, bounds in input_bounds.items()
    ]
    pd.DataFrame(bounds_rows).to_csv(
        output_directory / "sampling_bounds.csv", index=False
    )

    failed_mask = scenario_frame["Evaluation_Failed"] != 0
    successful = scenario_frame.loc[~failed_mask].copy()
    quantile_levels = tuple(QUANTILE_LEVELS)
    successful_output_columns = [
        *[
            f"{outcome_name}_{kind}_q{int(round(q * 100)):02d}"
            for kind in ("qr", "tskew")
            for q in quantile_levels
        ],
        *[
            f"{outcome_name}_tskew_{parameter}"
            for parameter in ("loc", "scale", "skew", "df")
        ],
        f"{outcome_name}_primary_tail_risk",
        f"{outcome_name}_qr_crossing_pair_count",
        f"{outcome_name}_qr_any_crossing",
    ]
    if outcome_name == "growth":
        successful_output_columns.append("growth_probability_below_zero")
    if not successful.empty and not np.isfinite(
        successful[successful_output_columns].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            f"A successful {narrative_id}/{outcome_name} EMA scenario "
            "has non-finite outputs."
        )

    summary_rows = []
    for prefix, label in (
        (f"{outcome_name}_qr_", "Direct quantile-regression estimate"),
        (f"{outcome_name}_tskew_", "Fitted T-skew quantile"),
    ):
        for quantile in quantile_levels:
            column = f"{prefix}q{int(round(quantile * 100)):02d}"
            values = successful[column] if not successful.empty else pd.Series(dtype=float)
            if not values.empty:
                summary_rows.append(
                    {
                        "outcome": column,
                        "description": f"{label} at {quantile:.0%}",
                        "minimum": float(values.min()),
                        "p05": float(values.quantile(0.05)),
                        "median": float(values.median()),
                        "p95": float(values.quantile(0.95)),
                        "maximum": float(values.max()),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(
        output_directory / "outcome_summary.csv", index=False
    )
    crossing_rows = [
        {
            "quantile_pair": (
                f"q{int(round(lower * 100)):02d}_"
                f"q{int(round(upper * 100)):02d}"
            ),
            "scenarios_with_crossing": int(
                sum(
                    successful[f"{outcome_name}_qr_q{int(round(lower * 100)):02d}"]
                    >= successful[
                        f"{outcome_name}_qr_q{int(round(upper * 100)):02d}"
                    ]
                )
            )
            if not successful.empty
            else 0,
            "successful_scenarios": len(successful),
        }
        for lower, upper in zip(quantile_levels[:-1], quantile_levels[1:])
    ]
    pd.DataFrame(crossing_rows).to_csv(
        output_directory / "quantile_crossing_diagnostics.csv",
        index=False,
    )

    outcome_spec = OUTCOME_SPECS[outcome_name]
    primary_quantile = float(outcome_spec["primary_quantile"])
    reference_row: dict[str, Any] = {
        **reference_inputs,
        "narrative_id": narrative_id,
        "outcome": outcome_name,
        "dependent_variable": outcome_spec["dependent_variable"],
        "forecast_origin": str(model.as_of.date()),
        "target_date": (
            model.as_of + pd.offsets.QuarterEnd(model.horizon)
        ).strftime("%Y-%m-%d"),
        "horizon_quarters": model.horizon,
        "training_observations": len(model.training_data),
        "primary_tail_quantile": primary_quantile,
        "primary_tail_risk": reference_prediction[
            "primary_distribution_quantile"
        ],
    }
    for quantile, value in reference_prediction["quantile_regression"].items():
        reference_row[
            f"qr_q{int(round(quantile * 100)):02d}"
        ] = value
    for quantile, value in reference_prediction["distribution_quantiles"].items():
        reference_row[
            f"tskew_q{int(round(quantile * 100)):02d}"
        ] = value
    reference_row.update(
        {
            f"tskew_{name}": value
            for name, value in reference_prediction[
                "distribution_parameters"
            ].items()
        }
    )
    pd.DataFrame([reference_row]).to_csv(
        output_directory / "reference_forecast.csv", index=False
    )

    if not successful.empty:
        _write_figures(
            successful,
            reference_prediction,
            output_directory,
            predictors,
            outcome_name,
        )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "analysis/run_ema_sampling.py",
        "narrative_id": narrative_id,
        "model_name": narrative_id,
        "model_label": MODEL_SPECS[narrative_id]["label"],
        "outcome": outcome_name,
        "dependent_variable": outcome_spec["dependent_variable"],
        "primary_risk_indicator": (
            f"{outcome_spec['risk_name']} fitted conditional "
            f"{primary_quantile:.0%} percentile"
        ),
        "predictors": list(predictors),
        "sampling_method": "EMA Workbench LHSSampler (Latin Hypercube)",
        "random_seed": RANDOM_SEED,
        "scenario_count_requested": N_SCENARIOS,
        "scenario_count_returned": len(scenario_frame),
        "input_range_percentiles": [
            EMA_INPUT_LOWER_PERCENTILE,
            EMA_INPUT_UPPER_PERCENTILE,
        ],
        "input_bound_sample": (
            "Each predictor's full available history in the raw BQR data; "
            "the same bounds are used for both outcomes regardless of their "
            "different aligned estimation samples."
        ),
        "input_bounds": _jsonify(bounds_rows),
        "reference_output": _jsonify(reference_row),
        "successful_scenarios": int((~failed_mask).sum()),
        "failed_scenarios": int(failed_mask.sum()),
        "scenarios_with_direct_quantile_crossing": int(
            successful[f"{outcome_name}_qr_any_crossing"].sum()
        )
        if not successful.empty
        else 0,
        "model_fit_calls": 1,
        "coefficients_fixed_across_scenarios": True,
        "training_observations": len(model.training_data),
        "training_target_start": str(model.training_data["target_date"].min().date()),
        "training_target_end": str(model.training_data["target_date"].max().date()),
        "information_cutoff": str(model.as_of.date()),
        "forecast_horizon_quarters": model.horizon,
        "independent_input_sampling": True,
        "scenario_frequency_interpretation": (
            "Exploratory design coverage, not a probability of future events."
        ),
        "assumptions": [
            "Empirical marginal input percentiles are provisional bounds, "
            "not validated economic uncertainty ranges.",
            "Inputs are sampled independently; combinations can violate "
            "historical economic dependencies.",
            "Regression coefficients are fitted once per outcome and remain "
            "fixed across scenarios; only conditional quantiles and the "
            "T-skew distribution are recomputed for scenario inputs.",
            "Within a narrative, both outcomes use one shared LHS design and "
            "the same predictor-history bounds.",
            "Combined growth and inflation results are separate conditional "
            "marginal distributions, not a joint probability distribution.",
        ],
    }
    with (output_directory / "run_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print(
        f"  {outcome_name}: {len(successful)} successful, "
        f"{int(failed_mask.sum())} failed; "
        f"training observations={len(model.training_data)}"
    )
    return scenario_frame


def _run_narrative_ema(
    model_name: str,
    raw_data: pd.DataFrame,
    output_root: Path,
    outcome_names: tuple[str, ...],
) -> pd.DataFrame:
    """Run one shared LHS design and evaluate selected outcomes for a narrative."""
    if EMA_SAMPLING_METHOD != "latin_hypercube":
        raise ValueError(
            "Unsupported EMA sampling method "
            f"{EMA_SAMPLING_METHOD!r}; use 'latin_hypercube'."
        )
    predictors = tuple(MODEL_SPECS[model_name]["predictors"])
    output_directory = output_root / MODEL_OUTPUT_NAMES[model_name]
    output_directory.mkdir(parents=True, exist_ok=True)

    models = {}
    references = {}
    for outcome_name in outcome_names:
        outcome_spec = OUTCOME_SPECS[outcome_name]
        model = fit_model_spec(
            model_name,
            raw_data,
            as_of=raw_data.index.max(),
            dependent_variable=outcome_spec["dependent_variable"],
        )
        if model.training_data is None or model.regression is None:
            raise RuntimeError(
                f"The {model_name}/{outcome_name} model did not fit successfully."
            )
        models[outcome_name] = model
        references[outcome_name] = {
            "inputs": {
                name: float(raw_data.iloc[-1][name]) for name in predictors
            },
            "prediction": model.predict(
                {
                    name: float(raw_data.iloc[-1][name])
                    for name in predictors
                }
            ),
        }

    # Calibrate once from full predictor histories, not outcome-specific
    # aligned estimation samples, so both outcomes share identical bounds.
    input_bounds = provisional_input_bounds(
        raw_data,
        predictors,
        lower_percentile=EMA_INPUT_LOWER_PERCENTILE,
        upper_percentile=EMA_INPUT_UPPER_PERCENTILE,
    )
    lower_column = _percentile_column("lower", EMA_INPUT_LOWER_PERCENTILE)
    upper_column = _percentile_column("upper", EMA_INPUT_UPPER_PERCENTILE)
    bounds = pd.DataFrame(
        [
            {
                "variable": name,
                lower_column: input_bounds[name][0],
                upper_column: input_bounds[name][1],
                "observations": int(raw_data[name].notna().sum()),
                "unit_note": (
                    "Prepared BQR column units; exact transformations/units "
                    "are not fully documented."
                ),
            }
            for name in predictors
        ]
    )
    bounds.to_csv(output_directory / "sampling_bounds.csv", index=False)

    failures: list[dict[str, Any]] = []
    outcome_names = tuple(outcome_names)
    output_names = [
        column
        for outcome_name in outcome_names
        for column in _outcome_columns(outcome_name)
    ] + [f"{outcome_name}_evaluation_failed" for outcome_name in outcome_names]

    def evaluate_scenario(**kwargs: float) -> dict[str, float]:
        """Evaluate every selected outcome at the same sampled inputs."""
        inputs = {name: float(kwargs[name]) for name in predictors}
        scenario_id = _scenario_identifier(model_name, inputs, predictors)
        scenario_outcomes: dict[str, float] = {}
        for outcome_name, model in models.items():
            try:
                prediction = model.predict(inputs)
            except Exception as error:
                failures.append(
                    {
                        "narrative_id": model_name,
                        "scenario_id": scenario_id,
                        "outcome": outcome_name,
                        **inputs,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                scenario_outcomes.update(
                    {
                        column: float("nan")
                        for column in _outcome_columns(outcome_name)
                    }
                )
                scenario_outcomes[f"{outcome_name}_evaluation_failed"] = 1.0
                continue
            scenario_outcomes.update(
                _prediction_outcomes(outcome_name, prediction)
            )
            scenario_outcomes[f"{outcome_name}_evaluation_failed"] = 0.0
        return scenario_outcomes

    ema_model = Model(f"MacroAtRisk_{model_name}", function=evaluate_scenario)
    ema_model.uncertainties = [
        RealParameter(
            row.variable,
            float(getattr(row, lower_column)),
            float(getattr(row, upper_column)),
        )
        for row in bounds.itertuples(index=False)
    ]
    ema_model.outcomes = [ScalarOutcome(name) for name in output_names]

    with SequentialEvaluator(ema_model) as evaluator:
        experiments, outcomes = perform_experiments(
            ema_model,
            scenarios=N_SCENARIOS,
            evaluator=evaluator,
            uncertainty_sampling=LHSSampler(),
            uncertainty_sampling_kwargs={"rng": RANDOM_SEED},
            log_progress=False,
        )

    if len(experiments) != N_SCENARIOS:
        raise RuntimeError(
            f"EMA Workbench returned {len(experiments)} experiments for "
            f"{model_name}; expected {N_SCENARIOS}."
        )
    scenario_results = pd.concat(
        [
            experiments.reset_index(drop=True),
            pd.DataFrame(outcomes).reset_index(drop=True),
        ],
        axis=1,
    )
    scenario_ids = [
        _scenario_identifier(
            model_name,
            {name: float(row[name]) for name in predictors},
            predictors,
        )
        for _, row in scenario_results.iterrows()
    ]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise RuntimeError(
            f"The {model_name} Latin Hypercube design contains duplicate "
            "input combinations."
        )
    scenario_results.insert(0, "scenario_id", scenario_ids)
    scenario_results.insert(0, "narrative_id", model_name)
    scenario_results["outcomes_included"] = ",".join(outcome_names)
    experiments = experiments.reset_index(drop=True).copy()
    experiments.insert(0, "scenario_id", scenario_ids)
    experiments.insert(0, "narrative_id", model_name)
    scenario_results.loc[
        :, ["narrative_id", "scenario_id", *predictors]
    ].to_csv(output_directory / "sampled_inputs.csv", index=False)
    scenario_results.to_csv(
        output_directory / "paired_scenario_results.csv", index=False
    )
    util.save_results(
        (experiments, outcomes),
        str(output_directory / "ema_workbench_results.tar.gz"),
    )
    for outcome_name, model in models.items():
        _write_outcome_results(
            scenario_results,
            failures,
            model,
            references[outcome_name]["prediction"],
            references[outcome_name]["inputs"],
            predictors,
            outcome_name,
            model_name,
            output_directory / outcome_name,
            input_bounds,
            {
                name: int(raw_data[name].notna().sum())
                for name in predictors
            },
        )

    combined = scenario_results.loc[
        :,
        [
            "narrative_id",
            "scenario_id",
            *predictors,
            *[
                column
                for column in scenario_results.columns
                if column.startswith(("growth_", "inflation_"))
            ],
            "outcomes_included",
        ],
    ].copy()
    combined.to_csv(output_directory / "combined_scenarios.csv", index=False)

    print(
        f"Narrative: {model_name}; predictors: {', '.join(predictors)}; "
        f"shared scenarios: {len(scenario_results)}"
    )
    print(f"Outputs written to: {output_directory}")
    return scenario_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("all", *MODEL_SEQUENCE),
        default="all",
        help="Narrative to explore; defaults to all three.",
    )
    parser.add_argument(
        "--outcome",
        choices=("all", *OUTCOME_SEQUENCE),
        default="all",
        help="Outcome family to explore; defaults to growth and inflation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="Root directory for narrative-specific EMA results.",
    )
    args = parser.parse_args()
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    raw_data = load_dnb_data(DATA_FILE, sheet_name=DATA_SHEET)
    model_names = MODEL_SEQUENCE if args.model == "all" else (args.model,)
    outcome_names = (
        OUTCOME_SEQUENCE if args.outcome == "all" else (args.outcome,)
    )
    for model_name in model_names:
        _run_narrative_ema(
            model_name,
            raw_data,
            output_root,
            outcome_names,
        )


if __name__ == "__main__":
    main()

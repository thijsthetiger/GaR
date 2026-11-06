"""Run a reproducible EMA Workbench input-sampling experiment for DNB GaR."""

from __future__ import annotations

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
    PREDICTORS,
    DNBGrowthAtRisk,
    load_dnb_data,
    prepare_horizon_data,
)
from GAR.tsfit.tskew import tskew_pdf  # noqa: E402


N_SCENARIOS = 100
RANDOM_SEED = 20261103
BASELINE_TOLERANCE = 1e-8
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "ema_sampling"
BASELINE_FORECAST_PATH = PROJECT_ROOT / "results" / "latest_forecast.csv"
OUTCOME_NAMES = (
    "QR_q05",
    "Tskew_GaR_q05",
    "Tskew_Median",
    "Probability_Negative",
    "Tskew_Location",
    "Tskew_Scale",
    "Tskew_Skew",
    "Tskew_DF",
    "Evaluation_Failed",
)


def _write_figures(
    scenarios: pd.DataFrame,
    baseline_prediction: dict[str, Any],
    baseline_inputs: dict[str, float],
    output_directory: Path,
) -> None:
    """Write the three requested figures from evaluated EMA outcomes."""
    gar = scenarios["Tskew_GaR_q05"]
    baseline_gar = baseline_prediction["distribution_quantiles"][0.05]

    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.hist(gar, bins=15, color="#4472C4", edgecolor="white")
    axis.axvline(
        baseline_gar,
        color="#C00000",
        linestyle="--",
        linewidth=2,
        label=f"Baseline GaR: {baseline_gar:.3f}%",
    )
    axis.set(
        xlabel="T-skew fifth percentile (percentage points)",
        ylabel="Number of sampled input scenarios",
        title="GaR outcomes across sampled input scenarios",
    )
    axis.legend()
    fig.savefig(output_directory / "figure1_gar_scenario_histogram.png", dpi=220)
    plt.close(fig)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, name in zip(axes.flat, PREDICTORS):
        axis.scatter(
            scenarios[name],
            gar,
            alpha=0.8,
            color="#4472C4",
            edgecolors="white",
            linewidths=0.4,
        )
        axis.axhline(baseline_gar, color="#C00000", linestyle="--", linewidth=1)
        axis.set_xlabel(f"{name} (BQR column units)")
        axis.set_ylabel("T-skew GaR (percentage points)")
        axis.set_title(f"{name} versus scenario GaR")
    figure.savefig(output_directory / "figure2_inputs_vs_gar.png", dpi=220)
    plt.close(figure)

    adverse = scenarios.loc[gar.idxmin()]
    favorable = scenarios.loc[gar.idxmax()]
    scenario_inputs = {
        "Baseline (2026Q1)": baseline_inputs,
        "Most adverse sampled": {
            name: float(adverse[name]) for name in PREDICTORS
        },
        "Most favorable sampled": {
            name: float(favorable[name]) for name in PREDICTORS
        },
    }
    scenario_predictions = {
        "Baseline (2026Q1)": baseline_prediction,
        "Most adverse sampled": {
            "distribution_parameters": {
                "loc": float(adverse["Tskew_Location"]),
                "scale": float(adverse["Tskew_Scale"]),
                "skew": float(adverse["Tskew_Skew"]),
                "df": int(adverse["Tskew_DF"]),
            },
            "distribution_quantiles": {0.05: float(adverse["Tskew_GaR_q05"])},
        },
        "Most favorable sampled": {
            "distribution_parameters": {
                "loc": float(favorable["Tskew_Location"]),
                "scale": float(favorable["Tskew_Scale"]),
                "skew": float(favorable["Tskew_Skew"]),
                "df": int(favorable["Tskew_DF"]),
            },
            "distribution_quantiles": {
                0.05: float(favorable["Tskew_GaR_q05"])
            },
        },
    }

    grid_min = min(
        float(prediction["distribution_quantiles"][0.05])
        for prediction in scenario_predictions.values()
    )
    grid_max = max(
        float(prediction["distribution_parameters"]["loc"])
        for prediction in scenario_predictions.values()
    )
    x_values = np.linspace(grid_min - 5, grid_max + 5, 600)
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = ("#1F4E79", "#C00000", "#70AD47")
    for (label, prediction), color in zip(scenario_predictions.items(), colors):
        parameters = prediction["distribution_parameters"]
        density = np.array(
            [tskew_pdf(float(x), **parameters) for x in x_values]
        )
        q05 = float(prediction["distribution_quantiles"][0.05])
        axis.plot(x_values, density, color=color, linewidth=2, label=label)
        axis.axvline(q05, color=color, linestyle="--", linewidth=1.5)
        axis.text(
            q05,
            float(density.max()),
            f" q05={q05:.2f}%",
            color=color,
            rotation=90,
            va="top",
            ha="right",
        )
    axis.set(
        xlabel="Year-on-year real GDP growth (%)",
        ylabel="T-skew probability density",
        title="Conditional growth distributions for selected inputs",
    )
    axis.legend()
    fig.savefig(
        output_directory / "figure3_conditional_distributions.png",
        dpi=220,
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    data_path = PROJECT_ROOT / "BQR data yoy.xlsx"
    raw_data = load_dnb_data(data_path, sheet_name="data_yoy")
    aligned_data = prepare_horizon_data(raw_data)

    model = DNBGrowthAtRisk()
    model.fit(aligned_data, as_of=raw_data.index.max())
    if model.training_data is None or model.regression is None:
        raise RuntimeError("The baseline GaR model did not fit successfully.")
    if len(model.training_data) != 217:
        raise RuntimeError(
            "Expected 217 baseline training observations, got "
            f"{len(model.training_data)}."
        )

    latest_date = raw_data.index.max()
    baseline_inputs = {
        name: float(raw_data.loc[latest_date, name]) for name in PREDICTORS
    }
    baseline_prediction = model.predict(baseline_inputs)

    if not BASELINE_FORECAST_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot verify the existing baseline forecast: "
            f"{BASELINE_FORECAST_PATH}"
        )
    existing_baseline = pd.read_csv(BASELINE_FORECAST_PATH).iloc[0]
    baseline_comparisons = {
        "QR_q05": (
            baseline_prediction["quantile_regression"][0.05],
            existing_baseline["qr_q05"],
        ),
        "Tskew_GaR_q05": (
            baseline_prediction["distribution_quantiles"][0.05],
            existing_baseline["tskew_q05_GaR"],
        ),
        "Tskew_Median": (
            baseline_prediction["distribution_quantiles"][0.50],
            existing_baseline["tskew_q50"],
        ),
        "Probability_Negative": (
            baseline_prediction["probability_growth_below_zero"],
            existing_baseline["probability_growth_below_zero"],
        ),
    }
    for output_name, (actual, expected) in baseline_comparisons.items():
        if not np.isclose(
            float(actual),
            float(expected),
            rtol=0.0,
            atol=BASELINE_TOLERANCE,
        ):
            raise RuntimeError(
                f"Fitted baseline {output_name}={actual!r} does not match "
                f"existing baseline={expected!r} within "
                f"{BASELINE_TOLERANCE}."
            )

    bounds = pd.DataFrame(
        [
            {
                "variable": name,
                "lower_p05": float(raw_data[name].quantile(0.05)),
                "upper_p95": float(raw_data[name].quantile(0.95)),
                "unit_note": (
                    "Prepared BQR column units; exact transformations/units "
                    "are not fully documented."
                ),
            }
            for name in PREDICTORS
        ]
    )
    bounds.to_csv(OUTPUT_DIRECTORY / "sampling_bounds.csv", index=False)

    failures: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []

    def evaluate_scenario(
        NF_YER: float,
        NF_URX: float,
        DF_LTN_NFCN: float,
        DR_PYR: float,
    ) -> dict[str, float]:
        inputs = {
            "NF_YER": float(NF_YER),
            "NF_URX": float(NF_URX),
            "DF_LTN_NFCN": float(DF_LTN_NFCN),
            "DR_PYR": float(DR_PYR),
        }
        try:
            prediction = model.predict(inputs)
        except Exception as error:
            failures.append(
                {
                    **inputs,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            failed_outcomes = {
                **{name: float("nan") for name in OUTCOME_NAMES[:-1]},
                "Evaluation_Failed": 1.0,
            }
            evaluation_records.append({**inputs, **failed_outcomes})
            return failed_outcomes

        parameters = prediction["distribution_parameters"]
        scenario_outcomes = {
            "QR_q05": float(prediction["quantile_regression"][0.05]),
            "Tskew_GaR_q05": float(
                prediction["distribution_quantiles"][0.05]
            ),
            "Tskew_Median": float(
                prediction["distribution_quantiles"][0.50]
            ),
            "Probability_Negative": float(
                prediction["probability_growth_below_zero"]
            ),
            "Tskew_Location": float(parameters["loc"]),
            "Tskew_Scale": float(parameters["scale"]),
            "Tskew_Skew": float(parameters["skew"]),
            "Tskew_DF": float(parameters["df"]),
            "Evaluation_Failed": 0.0,
        }
        evaluation_records.append({**inputs, **scenario_outcomes})
        return scenario_outcomes

    ema_model = Model("DNB_GaR_input_uncertainty", function=evaluate_scenario)
    ema_model.uncertainties = [
        RealParameter(
            row.variable,
            float(row.lower_p05),
            float(row.upper_p95),
        )
        for row in bounds.itertuples(index=False)
    ]
    ema_model.outcomes = [
        ScalarOutcome(name) for name in OUTCOME_NAMES
    ]

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
            f"EMA Workbench returned {len(experiments)} experiments; "
            f"expected {N_SCENARIOS}."
        )
    if len(evaluation_records) != len(experiments):
        raise RuntimeError(
            "The EMA experiment rows and model callback evaluations differ."
        )
    for name in PREDICTORS:
        returned_inputs = experiments[name].to_numpy(dtype=float)
        evaluated_inputs = np.array(
            [record[name] for record in evaluation_records], dtype=float
        )
        if not np.array_equal(returned_inputs, evaluated_inputs):
            raise RuntimeError(
                f"EMA input rows are misaligned with callback inputs for {name}."
            )
    for name in OUTCOME_NAMES:
        returned_outcomes = np.asarray(outcomes[name], dtype=float)
        evaluated_outcomes = np.array(
            [record[name] for record in evaluation_records], dtype=float
        )
        if not np.allclose(
            returned_outcomes,
            evaluated_outcomes,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"EMA outcome rows are misaligned for {name}."
            )
    scenario_results = pd.concat(
        [
            experiments.reset_index(drop=True),
            pd.DataFrame(outcomes).reset_index(drop=True),
        ],
        axis=1,
    )
    if len(scenario_results) != N_SCENARIOS:
        raise RuntimeError("EMA inputs and outcomes are not aligned.")
    scenario_results.insert(0, "scenario_id", np.arange(1, N_SCENARIOS + 1))
    scenario_results.to_csv(
        OUTPUT_DIRECTORY / "scenario_results.csv", index=False
    )
    util.save_results(
        (experiments, outcomes),
        str(OUTPUT_DIRECTORY / "ema_workbench_results.tar.gz"),
    )

    failed_mask = scenario_results["Evaluation_Failed"] != 0
    failed_scenarios = scenario_results.loc[failed_mask, "scenario_id"].tolist()
    if failures:
        failure_frame = pd.DataFrame(failures)
        input_columns = list(PREDICTORS)
        input_to_id = {
            tuple(float(value) for value in row[input_columns]): int(row.scenario_id)
            for row in scenario_results.to_dict("records")
        }
        failure_frame.insert(
            0,
            "scenario_id",
            [
                input_to_id[tuple(float(row[name]) for name in input_columns)]
                for _, row in failure_frame.iterrows()
            ],
        )
        failure_frame.to_csv(
            OUTPUT_DIRECTORY / "scenario_failures.csv", index=False
        )
    else:
        pd.DataFrame(
            columns=["scenario_id", *PREDICTORS, "error_type", "error_message"]
        ).to_csv(OUTPUT_DIRECTORY / "scenario_failures.csv", index=False)

    successful = scenario_results.loc[~failed_mask].copy()
    numeric_outputs = [
        "QR_q05",
        "Tskew_GaR_q05",
        "Tskew_Median",
        "Probability_Negative",
        "Tskew_Location",
        "Tskew_Scale",
        "Tskew_Skew",
        "Tskew_DF",
    ]
    if not np.isfinite(successful[numeric_outputs].to_numpy(dtype=float)).all():
        raise RuntimeError("A successful EMA scenario has non-finite outputs.")
    for row in bounds.itertuples(index=False):
        if not scenario_results[row.variable].between(
            row.lower_p05, row.upper_p95, inclusive="both"
        ).all():
            raise RuntimeError(
                f"EMA sampled values outside the bounds for {row.variable}."
            )

    baseline_row = {
        **baseline_inputs,
        "forecast_origin": latest_date.strftime("%Y-%m-%d"),
        "target_date": (
            latest_date + pd.offsets.QuarterEnd(model.horizon)
        ).strftime("%Y-%m-%d"),
        "horizon_quarters": model.horizon,
        "training_observations": len(model.training_data),
        "qr_q05": baseline_prediction["quantile_regression"][0.05],
        "tskew_q05_GaR": baseline_prediction["distribution_quantiles"][0.05],
        "tskew_q50": baseline_prediction["distribution_quantiles"][0.50],
        "probability_growth_below_zero": baseline_prediction[
            "probability_growth_below_zero"
        ],
    }
    pd.DataFrame([baseline_row]).to_csv(
        OUTPUT_DIRECTORY / "baseline_forecast.csv", index=False
    )

    metrics = {
        "Tskew_GaR_q05": "T-skew GaR (fifth percentile)",
        "QR_q05": "Direct quantile-regression fifth percentile",
        "Tskew_Median": "T-skew median",
        "Probability_Negative": "Probability of negative GDP growth",
    }
    summary_rows = []
    for column, label in metrics.items():
        values = successful[column]
        summary_rows.append(
            {
                "outcome": column,
                "description": label,
                "minimum": float(values.min()),
                "p05": float(values.quantile(0.05)),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "maximum": float(values.max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIRECTORY / "outcome_summary.csv", index=False)

    _write_figures(
        successful.reset_index(drop=True),
        baseline_prediction,
        baseline_inputs,
        OUTPUT_DIRECTORY,
    )

    try:
        import ema_workbench
    except ImportError as error:
        raise RuntimeError("Could not retrieve EMA Workbench version.") from error
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "analysis/run_ema_sampling.py",
        "ema_workbench_version": getattr(
            ema_workbench, "__version__", "unknown"
        ),
        "python_version": sys.version,
        "sampling_method": "EMA Workbench LHSSampler (Latin Hypercube)",
        "random_seed": RANDOM_SEED,
        "scenario_count_requested": N_SCENARIOS,
        "scenario_count_returned": len(experiments),
        "model_callback_evaluations": len(evaluation_records),
        "successful_scenarios": len(successful),
        "failed_scenarios": failed_scenarios,
        "model_fit_calls": 1,
        "training_observations": len(model.training_data),
        "information_cutoff": str(model.as_of.date()),
        "forecast_horizon_quarters": model.horizon,
        "independent_input_sampling": True,
        "sampling_bounds": bounds.to_dict(orient="records"),
        "baseline_comparison_tolerance": BASELINE_TOLERANCE,
        "baseline_comparisons": {
            key: {
                "fitted_model": float(actual),
                "existing_baseline_file": float(expected),
                "absolute_difference": abs(float(actual) - float(expected)),
            }
            for key, (actual, expected) in baseline_comparisons.items()
        },
        "assumptions": [
            "Empirical marginal 5th and 95th percentiles are provisional "
            "exploratory bounds, not validated economic uncertainty ranges.",
            "Inputs are sampled independently; combinations can violate "
            "historical economic dependencies and are not equally likely "
            "future states.",
            "The distribution of scenario-level GaR outcomes is not a "
            "calibrated predictive probability distribution.",
            "The fitted quantile-regression coefficients are fixed across "
            "scenarios; the existing predict method refits only the "
            "conditional T-skew distribution from scenario-specific "
            "conditional quantiles.",
        ],
    }
    with (OUTPUT_DIRECTORY / "run_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print(f"EMA Workbench version: {metadata['ema_workbench_version']}")
    print(f"GaR model fit calls: {metadata['model_fit_calls']}")
    print(f"EMA experiments returned: {len(experiments)}")
    print(f"Scenario callback evaluations: {len(evaluation_records)}")
    print(f"Successful scenarios: {len(successful)}")
    print(f"Failed scenarios: {len(failed_scenarios)}")
    print(f"Baseline T-skew GaR: {baseline_row['tskew_q05_GaR']:.6f}%")
    print(
        "Scenario T-skew GaR range: "
        f"{successful['Tskew_GaR_q05'].min():.6f}% to "
        f"{successful['Tskew_GaR_q05'].max():.6f}%"
    )
    print(f"Outputs written to: {OUTPUT_DIRECTORY}")

    if failed_scenarios:
        raise RuntimeError(
            "One or more EMA scenarios failed; see scenario_failures.csv: "
            + ", ".join(str(value) for value in failed_scenarios)
        )


if __name__ == "__main__":
    main()

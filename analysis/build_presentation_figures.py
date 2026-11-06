"""Build presentation figures and group summaries from saved GaR outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GAR.dnb_gar import (  # noqa: E402
    PREDICTORS,
    QUANTILES,
    DNBGrowthAtRisk,
    load_dnb_data,
    prepare_horizon_data,
)
from GAR.tsfit.tskew import tskew_cdf, tskew_pdf, tskew_ppf  # noqa: E402


RESULTS_DIRECTORY = PROJECT_ROOT / "results"
EMA_DIRECTORY = RESULTS_DIRECTORY / "ema_sampling"
OUTPUT_DIRECTORY = RESULTS_DIRECTORY / "presentation_figures"
BASELINE_FORECAST_PATH = RESULTS_DIRECTORY / "latest_forecast.csv"
SCENARIO_RESULTS_PATH = EMA_DIRECTORY / "scenario_results.csv"
TOLERANCE = 2e-6
QUANTILE_LEVELS = np.array(QUANTILES, dtype=float)
QUANTILE_LABELS = {
    0.05: "q05",
    0.10: "q10",
    0.25: "q25",
    0.50: "q50",
    0.75: "q75",
    0.90: "q90",
    0.95: "q95",
}
SCENARIO_OUTPUT_COLUMNS = {
    "QR_q05": "qr_q05",
    "Tskew_GaR_q05": "tskew_q05_GaR",
    "Tskew_Median": "tskew_q50",
    "Probability_Negative": "probability_growth_below_zero",
    "Tskew_Location": "tskew_loc",
    "Tskew_Scale": "tskew_scale",
    "Tskew_Skew": "tskew_skew",
    "Tskew_DF": "tskew_df",
}


def _parameters(row: pd.Series | dict[str, Any]) -> dict[str, float | int]:
    return {
        "loc": float(row["Tskew_Location"]),
        "scale": float(row["Tskew_Scale"]),
        "skew": float(row["Tskew_Skew"]),
        "df": int(row["Tskew_DF"]),
    }


def _baseline_parameters(row: pd.Series) -> dict[str, float | int]:
    return {
        "loc": float(row["tskew_loc"]),
        "scale": float(row["tskew_scale"]),
        "skew": float(row["tskew_skew"]),
        "df": int(row["tskew_df"]),
    }


def _verify_saved_outputs(
    baseline: pd.Series,
    scenarios: pd.DataFrame,
) -> dict[str, Any]:
    data = load_dnb_data(PROJECT_ROOT / "BQR data yoy.xlsx")
    prepared = prepare_horizon_data(data)
    model = DNBGrowthAtRisk().fit(prepared, as_of=data.index.max())
    if model.training_data is None or len(model.training_data) != 217:
        raise RuntimeError("The baseline model did not reproduce 217 observations.")

    baseline_inputs = {
        name: float(data.iloc[-1][name]) for name in PREDICTORS
    }
    baseline_prediction = model.predict(baseline_inputs)
    baseline_comparisons = {
        "qr_q05": baseline_prediction["quantile_regression"][0.05],
        "tskew_q05_GaR": baseline_prediction["distribution_quantiles"][0.05],
        "tskew_q50": baseline_prediction["distribution_quantiles"][0.50],
        "probability_growth_below_zero": baseline_prediction[
            "probability_growth_below_zero"
        ],
    }
    for column, actual in baseline_comparisons.items():
        if not np.isclose(
            float(actual), float(baseline[column]), rtol=0.0, atol=TOLERANCE
        ):
            raise RuntimeError(
                f"Reproduced baseline {column} differs from saved output."
            )

    maximum_scenario_difference = 0.0
    for scenario in scenarios.to_dict(orient="records"):
        prediction = model.predict(
            {name: float(scenario[name]) for name in PREDICTORS}
        )
        actuals = {
            "QR_q05": prediction["quantile_regression"][0.05],
            "Tskew_GaR_q05": prediction["distribution_quantiles"][0.05],
            "Tskew_Median": prediction["distribution_quantiles"][0.50],
            "Probability_Negative": prediction[
                "probability_growth_below_zero"
            ],
            "Tskew_Location": prediction["distribution_parameters"]["loc"],
            "Tskew_Scale": prediction["distribution_parameters"]["scale"],
            "Tskew_Skew": prediction["distribution_parameters"]["skew"],
            "Tskew_DF": prediction["distribution_parameters"]["df"],
        }
        for column, actual in actuals.items():
            saved = float(scenario[column])
            difference = abs(float(actual) - saved)
            maximum_scenario_difference = max(
                maximum_scenario_difference, difference
            )
            if not np.isclose(
                float(actual), saved, rtol=0.0, atol=TOLERANCE
            ):
                raise RuntimeError(
                    f"Scenario {scenario['scenario_id']} {column} does not "
                    "match the fitted model for its saved inputs."
                )

    return {
        "training_observations": len(model.training_data),
        "model_fit_calls": 1,
        "scenario_predict_calls": len(scenarios),
        "baseline_values_match_saved_forecast": True,
        "scenario_inputs_reproduce_saved_outputs": True,
        "maximum_scenario_output_absolute_difference": (
            maximum_scenario_difference
        ),
    }


def _integral(parameters: dict[str, float | int]) -> float:
    loc = float(parameters["loc"])
    left = quad(
        lambda value: tskew_pdf(value, **parameters),
        -np.inf,
        loc,
        epsabs=1e-8,
        epsrel=1e-8,
        limit=200,
    )[0]
    right = quad(
        lambda value: tskew_pdf(value, **parameters),
        loc,
        np.inf,
        epsabs=1e-8,
        epsrel=1e-8,
        limit=200,
    )[0]
    return float(left + right)


def _distribution_grid(
    distributions: list[dict[str, float | int]],
    points: int = 1600,
) -> np.ndarray:
    lower = min(
        float(tskew_ppf(0.05, **parameters))
        for parameters in distributions
    )
    upper = max(
        float(tskew_ppf(0.95, **parameters))
        for parameters in distributions
    )
    padding = max((upper - lower) * 0.08, 0.5)
    return np.linspace(lower - padding, upper + padding, points)


def _mixture_cdf(
    value: float,
    distributions: list[dict[str, float | int]],
) -> float:
    return float(
        np.mean([tskew_cdf(value, **parameters) for parameters in distributions])
    )


def _mixture_quantile(
    probability: float,
    distributions: list[dict[str, float | int]],
    component_quantiles: np.ndarray,
) -> tuple[float, float]:
    lower = float(component_quantiles.min())
    upper = float(component_quantiles.max())
    quantile = float(
        brentq(
            lambda value: _mixture_cdf(value, distributions) - probability,
            lower,
            upper,
            xtol=1e-12,
            rtol=1e-12,
        )
    )
    residual = abs(_mixture_cdf(quantile, distributions) - probability)
    return quantile, residual


def _save_quantile_fit_figure(
    baseline: pd.Series,
    baseline_parameters: dict[str, float | int],
    output_directory: Path,
) -> None:
    direct_quantiles = np.array(
        [
            float(baseline[f"qr_{QUANTILE_LABELS[float(tau)]}"])
            for tau in QUANTILE_LEVELS
        ]
    )
    grid = _distribution_grid([baseline_parameters])
    cdf_values = np.array(
        [tskew_cdf(value, **baseline_parameters) for value in grid]
    )
    fitted_quantiles = {
        tau: float(tskew_ppf(float(tau), **baseline_parameters))
        for tau in QUANTILE_LEVELS
    }

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.plot(
        grid,
        cdf_values,
        color="#1F4E79",
        linewidth=2.5,
        label="Fitted IMF T-skew CDF",
    )
    axis.scatter(
        direct_quantiles,
        QUANTILE_LEVELS,
        color="#ED7D31",
        edgecolor="white",
        linewidth=0.8,
        s=65,
        zorder=3,
        label="Seven direct QR quantiles",
    )
    for tau in (0.05, 0.50):
        x_value = fitted_quantiles[tau]
        axis.scatter(
            [x_value],
            [tau],
            marker="D",
            s=75,
            color="#C00000" if tau == 0.05 else "#70AD47",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label=f"T-skew q{int(tau * 100):02d}: {x_value:.3f}%",
        )
    axis.set(
        xlabel="Four-quarter-ahead year-on-year real GDP growth (%)",
        ylabel="Cumulative probability",
        title=(
            "Seven baseline QR estimates and fitted T-skew CDF "
            "(central 90% display range)"
        ),
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    figure.savefig(
        output_directory / "figure_quantiles_to_tskew_cdf.png", dpi=240
    )
    plt.close(figure)


def _save_baseline_cdf_figure(
    baseline: pd.Series,
    parameters: dict[str, float | int],
    output_directory: Path,
) -> None:
    grid = _distribution_grid([parameters])
    cdf_values = np.array([tskew_cdf(value, **parameters) for value in grid])
    q05 = float(baseline["tskew_q05_GaR"])
    q50 = float(baseline["tskew_q50"])
    negative_probability = float(
        baseline["probability_growth_below_zero"]
    )
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.plot(grid, cdf_values, color="#1F4E79", linewidth=2.5)
    axis.axvline(0, color="#7F7F7F", linestyle=":", linewidth=1.5)
    axis.axvline(q05, color="#C00000", linestyle="--", linewidth=1.5)
    axis.axvline(q50, color="#70AD47", linestyle="--", linewidth=1.5)
    axis.axhline(0.05, color="#C00000", linestyle=":", linewidth=1)
    axis.scatter(
        [0, q05, q50],
        [negative_probability, 0.05, 0.50],
        color=["#7F7F7F", "#C00000", "#70AD47"],
        zorder=3,
    )
    axis.annotate(
        f"Pr(growth < 0) = {negative_probability:.1%}",
        (0, negative_probability),
        xytext=(12, 12),
        textcoords="offset points",
    )
    axis.annotate(
        f"T-skew q05 = {q05:.3f}%",
        (q05, 0.05),
        xytext=(-12, 16),
        textcoords="offset points",
        ha="right",
        color="#C00000",
    )
    axis.annotate(
        f"Median = {q50:.3f}%",
        (q50, 0.50),
        xytext=(8, 10),
        textcoords="offset points",
        color="#548235",
    )
    axis.set(
        xlabel="Four-quarter-ahead year-on-year real GDP growth (%)",
        ylabel="Cumulative probability",
        title=(
            "Baseline conditional GDP-growth CDF (central 90% display range)"
        ),
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.2)
    figure.savefig(
        output_directory / "figure_baseline_conditional_cdf.png", dpi=240
    )
    plt.close(figure)


def _save_all_scenario_pdfs(
    scenarios: pd.DataFrame,
    baseline_parameters: dict[str, float | int],
    output_directory: Path,
) -> None:
    scenario_parameters = [
        _parameters(row)
        for row in scenarios.to_dict(orient="records")
    ]
    grid = _distribution_grid([*scenario_parameters, baseline_parameters])
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for parameters in scenario_parameters:
        densities = [tskew_pdf(value, **parameters) for value in grid]
        axis.plot(
            grid,
            densities,
            color="#4472C4",
            alpha=0.13,
            linewidth=0.8,
        )
    baseline_density = [
        tskew_pdf(value, **baseline_parameters) for value in grid
    ]
    axis.plot(
        grid,
        baseline_density,
        color="#C00000",
        linewidth=2.8,
        label="Observed baseline (2026Q1)",
        zorder=3,
    )
    axis.set(
        xlabel="Four-quarter-ahead year-on-year real GDP growth (%)",
        ylabel="T-skew probability density",
        title=(
            "100 EMA scenarios: conditional GDP-growth distributions "
            "(central 90% display range)"
        ),
    )
    axis.legend()
    axis.grid(alpha=0.18)
    figure.savefig(
        output_directory / "figure_100_scenario_conditional_pdfs.png",
        dpi=240,
    )
    plt.close(figure)


def _save_group_mixture_figures(
    baseline: pd.Series,
    scenarios: pd.DataFrame,
    baseline_parameters: dict[str, float | int],
    output_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ranked = scenarios.sort_values(
        ["Tskew_GaR_q05", "scenario_id"], ascending=[True, True]
    ).reset_index(drop=True)
    ranked["rank_by_GaR"] = np.arange(1, len(ranked) + 1)
    adverse = ranked.head(10).copy()
    favorable = ranked.tail(10).copy()
    if (
        adverse["scenario_id"].nunique() != 10
        or favorable["scenario_id"].nunique() != 10
        or set(adverse["scenario_id"]) & set(favorable["scenario_id"])
    ):
        raise RuntimeError("Adverse/favorable groups are not 10 distinct cases.")

    group_frames = {
        "Adverse 10 (lowest GaR)": adverse,
        "Favorable 10 (highest GaR)": favorable,
    }
    group_distributions = {
        name: [_parameters(row) for row in frame.to_dict(orient="records")]
        for name, frame in group_frames.items()
    }
    q05_components = {
        name: frame["Tskew_GaR_q05"].to_numpy(dtype=float)
        for name, frame in group_frames.items()
    }
    group_summary_rows: list[dict[str, Any]] = []
    quantile_residuals: dict[str, float] = {}
    for name, frame in group_frames.items():
        mixture_q05, residual = _mixture_quantile(
            0.05, group_distributions[name], q05_components[name]
        )
        quantile_residuals[name] = residual
        mean_individual_gar = float(frame["Tskew_GaR_q05"].mean())
        mixture_probability = _mixture_cdf(
            0.0, group_distributions[name]
        )
        mean_individual_probability = float(
            frame["Probability_Negative"].mean()
        )
        group_summary_rows.append(
            {
                "group": name,
                "scenario_count": len(frame),
                "scenario_ids": ";".join(
                    str(int(value)) for value in frame["scenario_id"]
                ),
                "defining_gar_minimum": float(
                    frame["Tskew_GaR_q05"].min()
                ),
                "defining_gar_maximum": float(
                    frame["Tskew_GaR_q05"].max()
                ),
                "mean_individual_gar": mean_individual_gar,
                "mixture_q05": mixture_q05,
                "mixture_q05_minus_mean_individual_gar": (
                    mixture_q05 - mean_individual_gar
                ),
                "mean_individual_probability_negative": (
                    mean_individual_probability
                ),
                "mixture_probability_negative": mixture_probability,
                "mixture_q05_cdf_residual": residual,
            }
        )

    summary = pd.DataFrame(group_summary_rows)
    members = pd.concat(
        [
            adverse.assign(group="Adverse 10 (lowest GaR)"),
            favorable.assign(group="Favorable 10 (highest GaR)"),
        ],
        ignore_index=True,
    )
    members.to_csv(
        output_directory / "adverse_favorable_scenario_members.csv",
        index=False,
    )
    summary.to_csv(
        output_directory / "adverse_favorable_group_summary.csv",
        index=False,
    )

    distributions = [
        baseline_parameters,
        *group_distributions["Adverse 10 (lowest GaR)"],
        *group_distributions["Favorable 10 (highest GaR)"],
    ]
    grid = _distribution_grid(distributions, points=2000)
    curve_map: dict[str, np.ndarray] = {
        "Observed baseline (2026Q1)": np.array(
            [tskew_pdf(value, **baseline_parameters) for value in grid]
        )
    }
    cdf_map: dict[str, np.ndarray] = {
        "Observed baseline (2026Q1)": np.array(
            [tskew_cdf(value, **baseline_parameters) for value in grid]
        )
    }
    for name, components in group_distributions.items():
        curve_map[name] = np.mean(
            [
                [tskew_pdf(value, **parameters) for value in grid]
                for parameters in components
            ],
            axis=0,
        )
        cdf_map[name] = np.mean(
            [
                [tskew_cdf(value, **parameters) for value in grid]
                for parameters in components
            ],
            axis=0,
        )

    colors = {
        "Observed baseline (2026Q1)": "#1F4E79",
        "Adverse 10 (lowest GaR)": "#C00000",
        "Favorable 10 (highest GaR)": "#70AD47",
    }
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name, density in curve_map.items():
        axis.plot(
            grid,
            density,
            color=colors[name],
            linewidth=2.5,
            label=name,
        )
    axis.set(
        xlabel="Four-quarter-ahead year-on-year real GDP growth (%)",
        ylabel="Probability density",
        title=(
            "Baseline and scenario-group PDF mixtures "
            "(central 90% display range)"
        ),
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(
        output_directory / "figure_adverse_favorable_mixture_pdfs.png",
        dpi=240,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.axhline(0.05, color="#7F7F7F", linestyle=":", linewidth=1.2)
    for name, cdf in cdf_map.items():
        axis.plot(
            grid,
            cdf,
            color=colors[name],
            linewidth=2.5,
            label=name,
        )
    for _, row in summary.iterrows():
        name = str(row["group"])
        axis.scatter(
            [float(row["mixture_q05"])],
            [0.05],
            color=colors[name],
            edgecolor="white",
            zorder=4,
        )
    axis.set(
        xlabel="Four-quarter-ahead year-on-year real GDP growth (%)",
        ylabel="Cumulative probability",
        title=(
            "Baseline and scenario-group CDF mixtures "
            "(central 90% display range)"
        ),
        ylim=(-0.02, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    figure.savefig(
        output_directory / "figure_adverse_favorable_mixture_cdfs.png",
        dpi=240,
    )
    plt.close(figure)

    baseline_q05 = float(baseline["tskew_q05_GaR"])
    baseline_probability = float(
        baseline["probability_growth_below_zero"]
    )
    baseline_summary = pd.DataFrame(
        [
            {
                "group": "Observed baseline (2026Q1)",
                "scenario_count": 1,
                "scenario_ids": "baseline",
                "defining_gar_minimum": baseline_q05,
                "defining_gar_maximum": baseline_q05,
                "mean_individual_gar": baseline_q05,
                "mixture_q05": baseline_q05,
                "mixture_q05_minus_mean_individual_gar": 0.0,
                "mean_individual_probability_negative": baseline_probability,
                "mixture_probability_negative": baseline_probability,
                "mixture_q05_cdf_residual": abs(
                    tskew_cdf(baseline_q05, **baseline_parameters) - 0.05
                ),
            }
        ]
    )
    report_summary = pd.concat(
        [baseline_summary, summary], ignore_index=True
    )
    report_summary.to_csv(
        output_directory / "presentation_distribution_summary.csv",
        index=False,
    )
    return members, summary, {
        "scenario_count": len(scenarios),
        "adverse_unique_scenarios": int(adverse["scenario_id"].nunique()),
        "favorable_unique_scenarios": int(
            favorable["scenario_id"].nunique()
        ),
        "groups_disjoint": not bool(
            set(adverse["scenario_id"]) & set(favorable["scenario_id"])
        ),
        "mixture_quantile_cdf_residuals": quantile_residuals,
    }


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    baseline_frame = pd.read_csv(BASELINE_FORECAST_PATH)
    if len(baseline_frame) != 1:
        raise RuntimeError("Expected exactly one saved baseline forecast row.")
    baseline = baseline_frame.iloc[0]
    scenarios = pd.read_csv(SCENARIO_RESULTS_PATH)
    if len(scenarios) != 100:
        raise RuntimeError(f"Expected 100 saved EMA scenarios, got {len(scenarios)}.")
    if scenarios["scenario_id"].nunique() != 100:
        raise RuntimeError("Saved EMA scenario identifiers are not unique.")
    if (scenarios["Evaluation_Failed"] != 0).any():
        raise RuntimeError("Saved EMA results include failed scenarios.")
    required = [
        *PREDICTORS,
        "scenario_id",
        "Tskew_GaR_q05",
        "Tskew_Median",
        "Probability_Negative",
        "Tskew_Location",
        "Tskew_Scale",
        "Tskew_Skew",
        "Tskew_DF",
    ]
    missing = [column for column in required if column not in scenarios]
    if missing:
        raise ValueError("Scenario results lack required columns: " + ", ".join(missing))

    verification = _verify_saved_outputs(baseline, scenarios)
    baseline_parameters = _baseline_parameters(baseline)
    scenario_parameters = [
        _parameters(row)
        for row in scenarios.to_dict(orient="records")
    ]
    distributions = [baseline_parameters, *scenario_parameters]
    density_integrals = [_integral(parameters) for parameters in distributions]
    if max(abs(value - 1.0) for value in density_integrals) > 2e-5:
        raise RuntimeError("At least one T-skew PDF does not integrate to one.")

    q05_checks = []
    for frame, parameters in [
        (pd.DataFrame([baseline]), baseline_parameters),
        (scenarios, None),
    ]:
        for row in frame.to_dict(orient="records"):
            distribution_parameters = (
                parameters if parameters is not None else _parameters(row)
            )
            actual_q05 = float(
                tskew_ppf(0.05, **distribution_parameters)
            )
            expected_q05 = float(
                row[
                    "tskew_q05_GaR"
                    if "tskew_q05_GaR" in row
                    else "Tskew_GaR_q05"
                ]
            )
            cdf_at_q05 = float(
                tskew_cdf(actual_q05, **distribution_parameters)
            )
            if (
                abs(actual_q05 - expected_q05) > TOLERANCE
                or abs(cdf_at_q05 - 0.05) > 1e-8
            ):
                raise RuntimeError("A saved T-skew fifth percentile failed validation.")
            q05_checks.append(
                {
                    "q05_absolute_difference": abs(
                        actual_q05 - expected_q05
                    ),
                    "cdf_at_q05": cdf_at_q05,
                }
            )

    grid = _distribution_grid(distributions)
    cdf_monotonicity_violations = 0
    for parameters in distributions:
        cdf_values = np.array(
            [tskew_cdf(value, **parameters) for value in grid]
        )
        cdf_monotonicity_violations += int(
            np.any(np.diff(cdf_values) < -1e-12)
        )
    if cdf_monotonicity_violations:
        raise RuntimeError("At least one T-skew CDF is not monotonic.")

    _save_quantile_fit_figure(
        baseline, baseline_parameters, OUTPUT_DIRECTORY
    )
    _save_baseline_cdf_figure(
        baseline, baseline_parameters, OUTPUT_DIRECTORY
    )
    _save_all_scenario_pdfs(
        scenarios, baseline_parameters, OUTPUT_DIRECTORY
    )
    members, group_summary, group_checks = _save_group_mixture_figures(
        baseline,
        scenarios,
        baseline_parameters,
        OUTPUT_DIRECTORY,
    )
    if len(members) != 20 or members["scenario_id"].nunique() != 20:
        raise RuntimeError("Group membership output does not contain 20 unique cases.")

    correlations = []
    for name in PREDICTORS:
        coefficient, p_value = spearmanr(
            scenarios[name], scenarios["Tskew_GaR_q05"]
        )
        correlations.append(
            {
                "predictor": name,
                "spearman_rho": float(coefficient),
                "p_value_descriptive": float(p_value),
                "interpretation_note": (
                    "Univariate association across this sampled design; "
                    "not causal and not a formal global sensitivity ranking."
                ),
            }
        )
    correlation_frame = pd.DataFrame(correlations)
    correlation_frame.to_csv(
        OUTPUT_DIRECTORY / "scenario_spearman_associations.csv",
        index=False,
    )

    metadata = {
        **verification,
        **group_checks,
        "scenario_rows_verified": len(scenarios),
        "distribution_count_validated": len(distributions),
        "pdf_integral_minimum": float(min(density_integrals)),
        "pdf_integral_maximum": float(max(density_integrals)),
        "maximum_pdf_integral_absolute_error": float(
            max(abs(value - 1.0) for value in density_integrals)
        ),
        "cdf_monotonicity_violations": cdf_monotonicity_violations,
        "maximum_q05_saved_value_absolute_difference": float(
            max(row["q05_absolute_difference"] for row in q05_checks)
        ),
        "maximum_q05_cdf_probability_absolute_difference": float(
            max(abs(row["cdf_at_q05"] - 0.05) for row in q05_checks)
        ),
        "baseline_q05": float(baseline["tskew_q05_GaR"]),
        "baseline_negative_growth_probability": float(
            baseline["probability_growth_below_zero"]
        ),
        "spearman_associations": correlation_frame.to_dict(orient="records"),
        "method_note": (
            "Conditional distributions use the saved scenario-specific IMF "
            "T-skew parameters and IMF T-skew PDF/CDF/PPF functions. "
            "The GaR quantile regressions are fitted once for alignment "
            "verification, not re-estimated per scenario."
        ),
    }
    with (OUTPUT_DIRECTORY / "presentation_figure_validation.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print(f"Validated {len(scenarios)} scenario-specific distributions.")
    print(
        "PDF integral range: "
        f"{metadata['pdf_integral_minimum']:.8f} to "
        f"{metadata['pdf_integral_maximum']:.8f}"
    )
    print(f"Adverse/favorable group sizes: 10 / 10")
    print(f"Outputs written to: {OUTPUT_DIRECTORY}")
    print(group_summary.to_string(index=False))


if __name__ == "__main__":
    main()

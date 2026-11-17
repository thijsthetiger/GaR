"""Fit the six thesis narrative Growth-at-Risk and Inflation-at-Risk models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    DATA_FILE,
    DATA_SHEET,
    MODEL_SEQUENCE,
    MODEL_OUTPUT_NAMES,
    MODEL_SPECS,
    NARRATIVE_MODEL_RESULTS_DIR,
    OUTCOME_SEQUENCE,
    OUTCOME_SPECS,
)
from GAR.dnb_gar import (
    fit_model_spec,
    load_dnb_data,
    prepare_horizon_data,
    write_model_results,
)


def main() -> None:
    """Fit selected narrative models and write model-specific outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("all", *MODEL_SEQUENCE),
        default="all",
        help="Narrative to fit; defaults to all three.",
    )
    parser.add_argument(
        "--outcome",
        choices=("all", *OUTCOME_SEQUENCE),
        default="all",
        help="Outcome family to fit; defaults to both growth and inflation.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_FILE,
        help="Workbook containing the configured BQR data worksheet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NARRATIVE_MODEL_RESULTS_DIR,
        help="Root directory for model-specific result folders.",
    )
    args = parser.parse_args()

    raw_data = load_dnb_data(args.data, sheet_name=DATA_SHEET)
    model_names = MODEL_SEQUENCE if args.model == "all" else (args.model,)
    outcome_names = OUTCOME_SEQUENCE if args.outcome == "all" else (args.outcome,)
    for model_name in model_names:
        predictors = MODEL_SPECS[model_name]["predictors"]
        for outcome_name in outcome_names:
            dependent_variable = OUTCOME_SPECS[outcome_name][
                "dependent_variable"
            ]
            prepared_data = prepare_horizon_data(
                raw_data,
                dependent_variable=dependent_variable,
                predictors=predictors,
            )
            model = fit_model_spec(
                model_name,
                raw_data,
                as_of=raw_data.index.max(),
                dependent_variable=dependent_variable,
            )
            if model.training_data is None:
                raise RuntimeError(
                    f"{model_name}/{outcome_name} did not retain fitted data."
                )
            output_directory = (
                args.output_dir
                / MODEL_OUTPUT_NAMES[model_name]
                / outcome_name
            )
            paths = write_model_results(
                model,
                raw_data,
                prepared_data,
                output_directory,
            )
            target_dates = model.training_data["target_date"]
            crossing_table = pd.read_csv(paths["crossing_diagnostics"])
            crossing_rows = int(
                crossing_table["rows_with_any_crossing"].iloc[0]
            )
            print(f"Model: {model_name}/{outcome_name}")
            print(f"Outcome: {dependent_variable}")
            print(f"Predictors: {', '.join(model.predictors)}")
            print(f"Estimation observations: {len(model.training_data)}")
            print(
                "Target-date sample: "
                f"{target_dates.min():%Y-%m-%d} to "
                f"{target_dates.max():%Y-%m-%d}"
            )
            print(f"Direct QR crossing rows (full aligned sample): {crossing_rows}")
            print(f"Latest forecast: {paths['latest_forecast']}")


if __name__ == "__main__":
    main()

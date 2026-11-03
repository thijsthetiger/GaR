"""Run the standalone DNB Growth-at-Risk model without Excel automation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from GAR.dnb_gar import (
    DNBGrowthAtRisk,
    load_dnb_data,
    prepare_horizon_data,
    write_model_results,
)


def main() -> None:
    """Load DNB data, estimate the model, and write numerical and plot results."""
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "BQR data yoy.xlsx",
        help="DNB workbook containing the data_yoy worksheet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results",
        help="Directory for CSV tables and PNG figures.",
    )
    args = parser.parse_args()

    raw_data = load_dnb_data(args.data)
    prepared_data = prepare_horizon_data(raw_data)
    model = DNBGrowthAtRisk()
    model.fit(prepared_data, as_of=raw_data.index.max())
    paths = write_model_results(model, raw_data, prepared_data, args.output_dir)

    latest = pd.read_csv(paths["latest_forecast"]).iloc[0]
    print(f"Estimation observations: {len(model.training_data)}")
    print(f"Information cutoff: {model.as_of:%Y-%m-%d}")
    print(f"Forecast origin: {latest['forecast_origin']}")
    print(f"Target date: {latest['target_date']}")
    print(f"QR 5th percentile: {latest['qr_q05']:.3f}%")
    print(f"T-skew GaR (5th percentile): {latest['tskew_q05_GaR']:.3f}%")
    print(f"Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

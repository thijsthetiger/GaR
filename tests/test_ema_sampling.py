"""Tests for paired growth/inflation EMA scenario handling."""

import numpy as np

from config import MODEL_SEQUENCE, MODEL_SPECS, OUTCOME_SEQUENCE, OUTCOME_SPECS
from analysis.run_ema_sampling import (
    _outcome_columns,
    _prediction_outcomes,
    _scenario_identifier,
)


def test_outcome_primary_tail_quantiles_are_configured():
    assert OUTCOME_SEQUENCE == ("growth", "inflation")
    for model_name in MODEL_SEQUENCE:
        assert isinstance(MODEL_SPECS[model_name]["predictors"], tuple)
        assert MODEL_SPECS[model_name]["predictors"][0] == "NF_YER"
        assert "DR_WDR" not in MODEL_SPECS[model_name]["predictors"]
    assert OUTCOME_SPECS["growth"]["primary_quantile"] == 0.05
    assert OUTCOME_SPECS["inflation"]["primary_quantile"] == 0.95


def test_scenario_identifier_is_stable_and_depends_on_inputs_and_narrative():
    predictors = ("NF_YER", "DR_FOD")
    inputs = {"NF_YER": 1.25, "DR_FOD": -0.5}

    scenario_id = _scenario_identifier("external_demand", inputs, predictors)

    assert scenario_id == _scenario_identifier(
        "external_demand",
        {"DR_FOD": -0.5, "NF_YER": 1.25},
        predictors,
    )
    assert scenario_id != _scenario_identifier(
        "external_demand",
        {"NF_YER": 1.25, "DR_FOD": -0.4},
        predictors,
    )
    assert scenario_id != _scenario_identifier(
        "financial_conditions", inputs, predictors
    )


def test_prediction_outcomes_retain_all_quantiles_and_primary_tail():
    quantiles = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    prediction = {
        "quantile_regression": {
            quantile: float(index) for index, quantile in enumerate(quantiles)
        },
        "distribution_quantiles": {
            quantile: float(index) + 0.25
            for index, quantile in enumerate(quantiles)
        },
        "distribution_parameters": {
            "loc": 1.0,
            "scale": 0.5,
            "skew": 1.1,
            "df": 2,
        },
        "primary_tail_quantile": 0.05,
        "primary_distribution_quantile": 0.30,
        "probability_below_zero": 0.2,
    }

    growth = _prediction_outcomes("growth", prediction)
    inflation_prediction = {
        **prediction,
        "primary_tail_quantile": 0.95,
        "primary_distribution_quantile": 6.25,
    }
    inflation = _prediction_outcomes("inflation", inflation_prediction)

    assert growth["growth_tskew_q05"] == 0.25
    assert growth["growth_tskew_q95"] == 6.25
    assert growth["growth_primary_tail_quantile"] == 0.05
    assert growth["growth_primary_tail_risk"] == 0.30
    assert growth["growth_probability_below_zero"] == 0.2
    assert growth["growth_qr_crossing_pair_count"] == 0
    assert growth["growth_qr_any_crossing"] == 0
    assert inflation["inflation_tskew_q05"] == 0.25
    assert inflation["inflation_tskew_q95"] == 6.25
    assert inflation["inflation_primary_tail_quantile"] == 0.95
    assert inflation["inflation_primary_tail_risk"] == 6.25
    assert inflation["inflation_qr_crossing_pair_count"] == 0
    assert inflation["inflation_qr_any_crossing"] == 0
    assert "inflation_probability_below_zero" not in inflation
    assert all(
        column in growth for column in _outcome_columns("growth")
    )
    assert all(
        column in inflation for column in _outcome_columns("inflation")
    )
    assert np.isfinite(list(growth.values())).all()
    assert np.isfinite(list(inflation.values())).all()

    crossed_prediction = {
        **prediction,
        "quantile_regression": {
            **prediction["quantile_regression"],
            0.95: -1.0,
        },
    }
    crossing_diagnostic = _prediction_outcomes("growth", crossed_prediction)
    assert crossing_diagnostic["growth_qr_crossing_pair_count"] == 1
    assert crossing_diagnostic["growth_qr_any_crossing"] == 1

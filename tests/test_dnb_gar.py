"""Focused tests for the thesis-specific DNB GaR adapter."""

import numpy as np
import pandas as pd
import pytest

from config import (
    DATA_FILE,
    DATA_SHEET,
    DEPENDENT_VARIABLES,
    EMA_INPUT_LOWER_PERCENTILE,
    EMA_INPUT_UPPER_PERCENTILE,
    EMA_RANDOM_SEED,
    EMA_SCENARIO_COUNT,
    FORECAST_HORIZON,
    MODEL_SEQUENCE as CONFIG_MODEL_SEQUENCE,
    MODEL_SPECS as CONFIG_MODEL_SPECS,
    OUTCOME_SEQUENCE,
    OUTCOME_SPECS,
    QR_CONFIDENCE_ALPHA,
    QR_MAX_ITER,
    QR_PARAMETER_TOLERANCE,
    QUANTILE_LEVELS,
)
from GAR.dnb_gar import (
    DEFAULT_HORIZON,
    DEFAULT_PREDICTORS,
    DNBGrowthAtRisk,
    MODEL_SEQUENCE,
    MODEL_SPECS,
    TARGET_COLUMN,
    TARGET_DATE_COLUMN,
    fit_model_spec,
    prepare_horizon_data,
    provisional_input_bounds,
)
from GAR.tsfit.tskew import tskew_cdf


def test_default_predictors_are_an_active_narrative_specification():
    assert DEFAULT_PREDICTORS == MODEL_SPECS["model_1_financial_conditions"][
        "predictors"
    ]


def test_central_configuration_preserves_current_model_and_ema_settings():
    assert DATA_FILE.name == "BQR data yoy.xlsx"
    assert DATA_SHEET == "data_yoy"
    assert DEPENDENT_VARIABLES == {"growth": "NF_YER", "inflation": "NF_HIC"}
    assert FORECAST_HORIZON == 4
    assert DEFAULT_HORIZON == FORECAST_HORIZON
    assert QUANTILE_LEVELS == (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    assert EMA_SCENARIO_COUNT == 100
    assert EMA_RANDOM_SEED == 20261103
    assert (EMA_INPUT_LOWER_PERCENTILE, EMA_INPUT_UPPER_PERCENTILE) == (0.05, 0.95)
    assert QR_CONFIDENCE_ALPHA == 0.1
    assert QR_MAX_ITER == 2000
    assert QR_PARAMETER_TOLERANCE == 1e-5
    assert MODEL_SEQUENCE == CONFIG_MODEL_SEQUENCE
    assert MODEL_SPECS is CONFIG_MODEL_SPECS
    assert MODEL_SPECS["model_1_financial_conditions"]["predictors"] == (
        "NF_YER",
        "DF_LTN_NFCN",
        "DF_STN_NFCN",
        "DF_STO",
    )
    assert MODEL_SPECS["model_2_external_demand"]["predictors"] == (
        "NF_YER",
        "DN_CXD",
        "DR_FOD",
    )
    assert MODEL_SPECS["model_3_household_housing"]["predictors"] == (
        "NF_YER",
        "DR_PYR",
        "DF_LTN_HHHN",
        "DF_IHX",
    )
    assert OUTCOME_SEQUENCE == ("growth", "inflation")
    assert OUTCOME_SPECS["growth"]["dependent_variable"] == "NF_YER"
    assert OUTCOME_SPECS["growth"]["primary_quantile"] == 0.05
    assert OUTCOME_SPECS["inflation"]["dependent_variable"] == "NF_HIC"
    assert OUTCOME_SPECS["inflation"]["primary_quantile"] == 0.95
    for spec in MODEL_SPECS.values():
        assert spec["predictors"][0] == "NF_YER"


def test_provisional_ema_bounds_use_configurable_percentiles():
    data = pd.DataFrame({"x": np.arange(101, dtype=float)})

    default_bounds = provisional_input_bounds(data, ("x",))
    narrower_bounds = provisional_input_bounds(
        data,
        ("x",),
        lower_percentile=0.25,
        upper_percentile=0.75,
    )

    assert default_bounds["x"] == pytest.approx((5, 95))
    assert narrower_bounds["x"] == pytest.approx((25, 75))
    with pytest.raises(ValueError, match="percentiles"):
        provisional_input_bounds(
            data,
            ("x",),
            lower_percentile=0.8,
            upper_percentile=0.2,
        )


def _quarterly_fixture(periods: int = 120) -> pd.DataFrame:
    """Build deterministic, non-collinear quarterly data for model tests."""
    rng = np.random.default_rng(2026)
    dates = pd.date_range("1995-03-31", periods=periods, freq="QE")
    predictors = pd.DataFrame(index=dates)
    for column in (
        *DEFAULT_PREDICTORS,
        "DF_STN_NFCN",
        "DF_STO",
        "DR_PYR",
        "DN_CXD",
        "DR_FOD",
        "DF_LTN_HHHN",
        "DF_IHX",
    ):
        predictors[column] = rng.normal(size=periods)
    growth = np.empty(periods)
    growth[0] = 2.0
    for index in range(1, periods):
        growth[index] = (
            0.65 * growth[index - 1]
            + 0.15 * predictors["DF_LTN_NFCN"].iloc[index]
            + rng.normal(scale=0.8)
        )
    predictors["NF_YER"] = growth
    predictors["DF_LTN_NFCN"] *= 0.5
    predictors["DR_PYR"] *= 1.5
    predictors["DF_STN_NFCN"] = 0.25 + predictors["DF_STN_NFCN"] * 0.4
    predictors["DF_STO"] = 100.0 + predictors["DF_STO"] * 12.0
    predictors["DN_CXD"] = 2.5 + predictors["DN_CXD"] * 0.4
    predictors["DR_FOD"] = 1.6 + predictors["DR_FOD"] * 0.9
    predictors["DF_LTN_HHHN"] = 0.5 + predictors["DF_LTN_HHHN"] * 0.9
    predictors["DF_IHX"] = 110.0 + predictors["DF_IHX"] * 5.0
    inflation = np.empty(periods)
    inflation[0] = 2.0
    for index in range(1, periods):
        inflation[index] = (
            0.4 * inflation[index - 1]
            + 0.2 * growth[index]
            + rng.normal(scale=0.7)
        )
    predictors["NF_HIC"] = inflation
    return predictors


def test_prepare_horizon_data_pairs_outcome_with_four_quarter_origin():
    data = _quarterly_fixture(periods=12)

    prepared = prepare_horizon_data(data, horizon=FORECAST_HORIZON)

    assert prepared.index[0] == data.index[0]
    assert prepared.index[-1] == data.index[-(FORECAST_HORIZON + 1)]
    assert (
        prepared.loc[data.index[0], TARGET_COLUMN]
        == data["NF_YER"].iloc[FORECAST_HORIZON]
    )
    assert (
        prepared.loc[data.index[0], TARGET_DATE_COLUMN]
        == data.index[FORECAST_HORIZON]
    )
    assert len(prepared) == len(data) - FORECAST_HORIZON


def test_prepare_horizon_data_aligns_inflation_without_changing_predictors():
    data = _quarterly_fixture(periods=12)
    predictors = MODEL_SPECS["model_2_external_demand"]["predictors"]

    prepared = prepare_horizon_data(
        data,
        horizon=FORECAST_HORIZON,
        dependent_variable=DEPENDENT_VARIABLES["inflation"],
        predictors=predictors,
    )

    assert "future_NF_HIC" in prepared
    assert prepared.loc[data.index[0], "future_NF_HIC"] == data["NF_HIC"].iloc[4]
    assert prepared.loc[data.index[0], TARGET_DATE_COLUMN] == data.index[4]
    assert tuple(name for name in predictors if name in prepared.columns) == predictors
    assert "DR_WDR" not in predictors


def test_fit_respects_information_cutoff_and_predicts_finite_ga_risk():
    data = _quarterly_fixture()
    prepared = prepare_horizon_data(data)
    cutoff = data.index[-5]

    model = DNBGrowthAtRisk().fit(prepared, as_of=cutoff)

    assert (model.training_data[TARGET_DATE_COLUMN] <= cutoff).all()
    assert len(model.training_data) < len(prepared)
    assert model.regression is not None
    assert model.regression.max_iter == QR_MAX_ITER
    assert model.regression.parameter_tolerance == QR_PARAMETER_TOLERANCE
    prediction = model.predict(
        data.loc[data.index[-1], list(DEFAULT_PREDICTORS)].to_dict()
    )
    assert np.isfinite(prediction["quantile_regression"][0.05])
    assert np.isfinite(prediction["distribution_quantiles"][0.05])
    assert prediction["distribution_parameters"]["scale"] > 0
    assert prediction["distribution_parameters"]["skew"] > 0
    assert 0 <= prediction["probability_growth_below_zero"] <= 1
    assert tskew_cdf(
        prediction["distribution_quantiles"][0.05],
        **prediction["distribution_parameters"],
    ) == pytest.approx(0.05, abs=1e-8)


def test_predict_requires_a_fitted_model_and_all_predictors():
    model = DNBGrowthAtRisk()
    with pytest.raises(RuntimeError, match="Call fit"):
        model.predict({})


def test_prepare_horizon_data_rejects_nonpositive_horizon():
    with pytest.raises(ValueError, match="positive integer"):
        prepare_horizon_data(_quarterly_fixture(), horizon=0)


def test_thesis_narrative_model_specs_exist_and_are_fittable():
    assert MODEL_SEQUENCE == (
        "model_1_financial_conditions",
        "model_2_external_demand",
        "model_3_household_housing",
    )
    assert set(MODEL_SPECS) == set(MODEL_SEQUENCE)
    for name in MODEL_SEQUENCE:
        spec = MODEL_SPECS[name]
        assert "NF_YER" in spec["predictors"]
        assert len(spec["predictors"]) in (3, 4)
        assert spec["narrative"]

    fixture = _quarterly_fixture()
    for name in MODEL_SEQUENCE:
        predictors = MODEL_SPECS[name]["predictors"]
        for outcome_spec in OUTCOME_SPECS.values():
            model = fit_model_spec(
                name,
                fixture,
                as_of=fixture.index[-1],
                dependent_variable=outcome_spec["dependent_variable"],
            )
            assert model.training_data is not None
            assert model.predictors == predictors
            assert model.target_column == f"future_{outcome_spec['dependent_variable']}"
            prediction = model.predict(
                {
                    column: float(fixture.iloc[-1][column])
                    for column in model.predictors
                }
            )
            assert set(prediction["quantile_regression"]) == set(QUANTILE_LEVELS)
            assert set(prediction["distribution_quantiles"]) == set(QUANTILE_LEVELS)
            assert np.isfinite(
                list(prediction["quantile_regression"].values())
            ).all()
            assert np.isfinite(
                list(prediction["distribution_quantiles"].values())
            ).all()
            assert prediction["distribution_parameters"]["scale"] > 0
            assert prediction["distribution_parameters"]["skew"] > 0
            assert prediction["primary_tail_quantile"] == outcome_spec[
                "primary_quantile"
            ]
            assert np.diff(list(prediction["distribution_quantiles"].values())).min() > 0

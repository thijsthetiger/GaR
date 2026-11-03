"""Focused tests for the thesis-specific DNB GaR adapter."""

import numpy as np
import pandas as pd
import pytest

from GAR.dnb_gar import (
    DNBGrowthAtRisk,
    PREDICTORS,
    TARGET_COLUMN,
    TARGET_DATE_COLUMN,
    prepare_horizon_data,
)
from GAR.tsfit.tskew import tskew_cdf


def test_model_uses_the_current_dnb_predictor_specification():
    assert PREDICTORS == ("NF_YER", "NF_URX", "DF_LTN_NFCN", "DR_PYR")


def _quarterly_fixture(periods: int = 120) -> pd.DataFrame:
    """Build deterministic quarterly data for model and alignment tests."""
    rng = np.random.default_rng(2026)
    dates = pd.date_range("1995-03-31", periods=periods, freq="QE")
    predictors = pd.DataFrame(
        rng.normal(size=(periods, len(PREDICTORS))),
        columns=PREDICTORS,
        index=dates,
    )
    predictors["NF_YER"] = 2.0 + predictors["NF_YER"]
    predictors["NF_URX"] = 6.0 + predictors["NF_URX"]
    predictors["DF_LTN_NFCN"] = predictors["DF_LTN_NFCN"] * 0.5
    predictors["DR_PYR"] = predictors["DR_PYR"] * 1.5
    predictors["NF_YER"] = (
        predictors["NF_YER"]
        + 0.3 * predictors["NF_URX"]
        - 0.2 * predictors["DF_LTN_NFCN"]
        + rng.normal(scale=1.2, size=periods)
    )
    return predictors


def test_prepare_horizon_data_pairs_outcome_with_four_quarter_origin():
    data = _quarterly_fixture(periods=12)

    prepared = prepare_horizon_data(data, horizon=4)

    assert prepared.index[0] == data.index[0]
    assert prepared.index[-1] == data.index[-5]
    assert prepared.loc[data.index[0], TARGET_COLUMN] == data["NF_YER"].iloc[4]
    assert prepared.loc[data.index[0], TARGET_DATE_COLUMN] == data.index[4]
    assert len(prepared) == len(data) - 4


def test_fit_respects_information_cutoff_and_predicts_finite_ga_risk():
    data = _quarterly_fixture()
    prepared = prepare_horizon_data(data)
    cutoff = data.index[-5]

    model = DNBGrowthAtRisk().fit(prepared, as_of=cutoff)

    assert (model.training_data[TARGET_DATE_COLUMN] <= cutoff).all()
    assert len(model.training_data) < len(prepared)
    prediction = model.predict(data.loc[data.index[-1], list(PREDICTORS)].to_dict())
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

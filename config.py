"""Editable project settings for the thesis Growth-at-Risk analyses."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Input data
DATA_FILE = PROJECT_ROOT / "BQR data yoy.xlsx"
DATA_SHEET = "data_yoy"
GROWTH_VARIABLE = "NF_YER"
INFLATION_VARIABLE = "NF_HIC"
DEPENDENT_VARIABLES = {
    "growth": GROWTH_VARIABLE,
    "inflation": INFLATION_VARIABLE,
}
FORECAST_HORIZON = 4

# Change predictor lists here to configure each narrative model.
MODEL_SPECS = {
    "model_1_financial_conditions": {
        "label": "Tightening financial and credit conditions",
        "predictors": ("NF_YER", "DF_LTN_NFCN", "DF_STN_NFCN", "DF_STO"),
        "narrative": (
            "Higher financing costs, tighter credit conditions and weaker "
            "equity valuations may affect future output and inflation risks."
        ),
        "candidate_variables": (
            "DF_LTN_NFCN",
            "DF_STN_NFCN",
            "DF_STO",
        ),
    },
    "model_2_external_demand": {
        "label": "External demand and competitiveness",
        "predictors": ("NF_YER", "DN_CXD", "DR_FOD"),
        "narrative": (
            "External demand and competitiveness conditions affect export-"
            "driven production, output, and inflation."
        ),
        "candidate_variables": ("DN_CXD", "DR_FOD"),
    },
    "model_3_household_housing": {
        "label": "Household demand and housing market conditions",
        "predictors": ("NF_YER", "DR_PYR", "DF_LTN_HHHN", "DF_IHX"),
        "narrative": (
            "Weaker disposable income, higher mortgage financing costs and "
            "cooling housing markets can affect demand, output, and inflation."
        ),
        "candidate_variables": ("DR_PYR", "DF_LTN_HHHN", "DF_IHX"),
    },
}
MODEL_SEQUENCE = tuple(MODEL_SPECS)
MODEL_OUTPUT_NAMES = {
    "model_1_financial_conditions": "financial_conditions",
    "model_2_external_demand": "external_demand",
    "model_3_household_housing": "household_housing",
}

# Each outcome is estimated separately with the narrative's unchanged inputs.
OUTCOME_SPECS = {
    "growth": {
        "dependent_variable": GROWTH_VARIABLE,
        "label": "GDP growth",
        "unit": "Year-on-year real GDP growth (%)",
        "risk_name": "Growth-at-Risk",
        "primary_quantile": 0.05,
        "tail_direction": "downside",
    },
    "inflation": {
        "dependent_variable": INFLATION_VARIABLE,
        "label": "HICP inflation",
        "unit": "NF_HIC units as supplied in data_yoy",
        "risk_name": "Inflation-at-Risk",
        "primary_quantile": 0.95,
        "tail_direction": "upside",
    },
}
OUTCOME_SEQUENCE = tuple(OUTCOME_SPECS)

# IMF quantile-regression and conditional T-skew settings.
QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
TSKEW_REQUIRED_QUANTILES = (0.05, 0.25, 0.50, 0.75)
QR_CONFIDENCE_ALPHA = 0.1
QR_MAX_ITER = 2000
QR_PARAMETER_TOLERANCE = 1e-5

# Preliminary EMA Workbench input-sampling settings.
EMA_SCENARIO_COUNT = 100
EMA_RANDOM_SEED = 20261103
EMA_SAMPLING_METHOD = "latin_hypercube"
EMA_INPUT_LOWER_PERCENTILE = 0.05
EMA_INPUT_UPPER_PERCENTILE = 0.95

# Output locations for active model estimates and EMA experiments.
RESULTS_DIR = PROJECT_ROOT / "results"
NARRATIVE_MODEL_RESULTS_DIR = RESULTS_DIR / "narrative_models"
EMA_RESULTS_DIR = RESULTS_DIR / "ema_sampling"

"""Step 2 - feature engineering as a reusable scikit-learn pipeline.

The notebooks applied pickled PowerTransformers / StandardScaler / OneHotEncoder
by hand and rebuilt the interaction terms in every notebook. Here the same
steps live inside one Pipeline, so training, evaluation and the web app all
transform data identically (no train/serve skew, one file to deploy).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from src.config import CATEGORICAL_FEATURES, MODEL_FEATURES, SKEWED_FEATURES, WEATHER_FEATURES

NUMERIC_PLAIN = [c for c in WEATHER_FEATURES if c not in SKEWED_FEATURES]

# Interaction terms from V2-V4 / Ridge notebooks (on transformed + scaled values)
INTERACTIONS = {
    "FFMC_TEMP": ("fine_fuel_moisture_code", "temperature"),
    "FFMC_RH": ("fine_fuel_moisture_code", "relative_humidity"),
    "RH_TEMP": ("relative_humidity", "temperature"),
    "RD_FFMC": ("rain_days", "fine_fuel_moisture_code"),
    "DC_RD": ("drought_code", "rain_days"),
    "WS_FFMC": ("wind_speed", "fine_fuel_moisture_code"),
    "ISI_WS": ("initial_spread_index", "wind_speed"),
    "ISI_FFMC": ("initial_spread_index", "fine_fuel_moisture_code"),
    "ISI_RH": ("initial_spread_index", "relative_humidity"),
    "DMC_TEMP": ("duff_moisture_code", "temperature"),
    "DMC_RD": ("duff_moisture_code", "rain_days"),
    "RD_TEMP": ("rain_days", "temperature"),
    "DMC_DC": ("duff_moisture_code", "drought_code"),
    "BUI_DC": ("build_up_index", "drought_code"),
    "BUI_DMC": ("build_up_index", "duff_moisture_code"),
    "ISI_BUI": ("initial_spread_index", "build_up_index"),
}
SQRT_TERMS = ["build_up_index", "fine_fuel_moisture_code", "initial_spread_index"]


def base_preprocessor() -> ColumnTransformer:
    """Yeo-Johnson on skewed columns, scaling on the rest, one-hot on country."""
    return ColumnTransformer(
        [
            ("skewed", Pipeline([("power", PowerTransformer(method="yeo-johnson")),
                                 ("scale", StandardScaler())]), SKEWED_FEATURES),
            ("plain", StandardScaler(), NUMERIC_PLAIN),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Adds pairwise products and sqrt terms (non-linearity for linear models)."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "n_features_in_")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for name, (a, b) in INTERACTIONS.items():
            X[name] = X[a] * X[b]
        for c in SQRT_TERMS:
            X[f"{c}_sqrt"] = np.sqrt(X[c].clip(lower=0))
        return X



def linear_features(with_interactions: bool = True) -> Pipeline:
    steps = [("prep", base_preprocessor())]
    if with_interactions:
        steps.append(("interactions", InteractionFeatures()))
    return Pipeline(steps)


def tree_features() -> ColumnTransformer:
    """Trees don't need scaling; only encode country."""
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        remainder="passthrough", verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def split_xy(df: pd.DataFrame, target="fire_weather_index"):
    return df[MODEL_FEATURES], df[target].astype(float)

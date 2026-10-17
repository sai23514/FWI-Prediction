"""Upgrades 1 + 2 - weather-only model and honest validation.

    python -m src.weather_features      # build data/processed/weather_only.parquet
    python -m src.advanced_validation   # train + evaluate, writes reports/advanced_results.json

Compares two models under three validation schemes:
  * full-input model   : 13 inputs incl. FFMC/DMC/DC/BUI/ISI (the deployed model)
  * weather-only model : weather + rolling weather history + season + location,
                         NO FWI-system codes
  schemes: random split | time split (train 2000-2007, test 2008-2009) |
           station holdout (20% of stations never seen in training)
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (FIGURES_DIR, MODEL_FEATURES, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE,
                        REPORTS_DIR, TARGET)
from src.model_development import metrics
from src.model_evaluation import danger_class
from src.weather_features import WEATHER_ONLY_FEATURES


def hgb():
    return Pipeline([
        ("features", ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["country"])],
            remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")),
        ("model", HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=127,
                                                max_iter=800, early_stopping=True,
                                                validation_fraction=0.1,
                                                random_state=RANDOM_STATE)),
    ])


def splits(df):
    tr, te = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE)
    yield "random_split", tr, te
    years = df["report_date"].dt.year
    yield "time_split_2008_2009", df.index[years <= 2007], df.index[years >= 2008]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(gss.split(df, groups=df["wmo"]))
    yield "unseen_stations", df.index[tr], df.index[te]


def evaluate(name, df, features):
    res = {}
    for scheme, tr, te in splits(df):
        t = time.time()
        m = hgb().fit(df.loc[tr, features], df.loc[tr, TARGET])
        p = np.clip(m.predict(df.loc[te, features]), 0, None)
        y = df.loc[te, TARGET].astype(float)
        r = {k: round(float(v), 4) for k, v in metrics(y, p).items()}
        r["danger_class_accuracy"] = round(float((danger_class(p) == danger_class(y)).mean()), 4)
        r["test_rows"] = int(len(te))
        res[scheme] = r
        print(f"{name:13s} {scheme:22s} {r}  ({time.time()-t:.0f}s)")
        if scheme == "time_split_2008_2009":
            last_model, last_test = m, (df.loc[te, features], y, p)
    return res, last_model, last_test


def main():
    full = pd.concat([pd.read_parquet(PROCESSED_DIR / f"{s}.parquet")
                      for s in ("train", "dev", "test")], ignore_index=True)
    weather = pd.read_parquet(PROCESSED_DIR / "weather_only.parquet").reset_index(drop=True)

    results = {}
    results["full_input"], _, _ = evaluate("full-input", full, MODEL_FEATURES)
    results["weather_only"], wmodel, (Xte, yte, pte) = evaluate("weather-only", weather,
                                                               WEATHER_ONLY_FEATURES)

    # final weather-only model on all data, for reuse
    final = hgb().fit(weather[WEATHER_ONLY_FEATURES], weather[TARGET])
    joblib.dump(final, MODELS_DIR / "fwi_weather_only.joblib", compress=3)

    # plots
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    s = rng.choice(len(yte), size=min(20000, len(yte)), replace=False)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(yte.iloc[s], pte[s], s=3, alpha=0.3)
    lim = [0, float(max(yte.max(), pte.max()))]
    ax.plot(lim, lim, "r--", lw=1)
    r = results["weather_only"]["time_split_2008_2009"]
    ax.set(title=f"Weather-only model, unseen years 2008-09\nR² {r['R2']:.3f}, MAE {r['MAE']:.2f}",
           xlabel="Actual FWI", ylabel="Predicted FWI")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "weather_only_actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    sub = rng.choice(len(yte), size=5000, replace=False)
    imp = permutation_importance(wmodel, Xte.iloc[sub], yte.iloc[sub], n_repeats=3,
                                 random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
    imp_s = pd.Series(imp.importances_mean, index=Xte.columns).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp_s.plot.barh(ax=ax, color="#2b6cd9")
    ax.set(title="Weather-only model - top 15 features (permutation importance)",
           xlabel="Increase in MAE")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "weather_only_feature_importance.png", dpi=120)
    plt.close(fig)
    results["weather_only_top_features"] = imp_s.sort_values(ascending=False).round(4).to_dict()

    (REPORTS_DIR / "advanced_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

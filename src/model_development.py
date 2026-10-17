"""Step 3 - train and compare models, keep the best one.

    python -m src.model_development

Candidates (all predict FWI on its original scale; linear models learn on a
Yeo-Johnson transformed target, as in the notebooks):
  1. Linear Regression, raw features         (LR_v1 baseline)
  2. Ridge + interaction terms, tuned alpha  (Ridge_Regression, the notebook "final")
  3. Random Forest                            (README future-work item)
  4. Histogram Gradient Boosting, tuned on the dev set
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

from src.config import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR
from src.feature_engineering import linear_features, split_xy, tree_features

RF_TRAIN_ROWS = 250_000  # Random Forest memory/time budget on a laptop
MAX_DEPLOY_MB = 50  # GitHub rejects files > 100 MB; keep the served model small


def with_power_target(model):
    return TransformedTargetRegressor(regressor=model,
                                      transformer=PowerTransformer(method="yeo-johnson"))


def metrics(y, p):
    p = np.clip(p, 0, None)  # FWI can't be negative
    return {"MAE": mean_absolute_error(y, p), "MSE": mean_squared_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": r2_score(y, p)}


def candidates():
    return {
        "linear_regression": Pipeline([("features", linear_features(False)),
                                       ("model", with_power_target(LinearRegression()))]),
        "ridge_interactions": Pipeline([("features", linear_features(True)),
                                        ("model", with_power_target(
                                            RidgeCV(alphas=np.logspace(-3, 3, 25))))]),
        "random_forest": Pipeline([("features", tree_features()),
                                   ("model", RandomForestRegressor(
                                       n_estimators=100, min_samples_leaf=2, max_features=0.6,
                                       n_jobs=-1, random_state=RANDOM_STATE))]),
    }


def tune_hgb(X_tr, y_tr, X_dev, y_dev):
    grid = [dict(learning_rate=lr, max_leaf_nodes=leaves, max_iter=600)
            for lr in (0.05, 0.1) for leaves in (31, 63, 127)]
    best, best_mae, log = None, np.inf, []
    for params in grid:
        m = Pipeline([("features", tree_features()),
                      ("model", HistGradientBoostingRegressor(
                          loss="squared_error", early_stopping=True, validation_fraction=0.1,
                          random_state=RANDOM_STATE, **params))])
        m.fit(X_tr, y_tr)
        mae = mean_absolute_error(y_dev, np.clip(m.predict(X_dev), 0, None))
        log.append({**params, "dev_MAE": round(mae, 4)})
        print(f"  HGB {params} dev MAE={mae:.4f}")
        if mae < best_mae:
            best, best_mae = m, mae
    return best, log


def main():
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    dev = pd.read_parquet(PROCESSED_DIR / "dev.parquet")
    X_tr, y_tr = split_xy(train)
    X_dev, y_dev = split_xy(dev)

    results, fitted = {}, {}
    for name, model in candidates().items():
        t = time.time()
        if name == "random_forest":
            idx = train.sample(n=min(RF_TRAIN_ROWS, len(train)), random_state=RANDOM_STATE).index
            model.fit(X_tr.loc[idx], y_tr.loc[idx])
        else:
            model.fit(X_tr, y_tr)
        results[name] = {"dev": metrics(y_dev, model.predict(X_dev)),
                         "train_seconds": round(time.time() - t, 1)}
        fitted[name] = model
        print(f"{name}: {results[name]}")

    t = time.time()
    hgb, hgb_log = tune_hgb(X_tr, y_tr, X_dev, y_dev)
    results["hist_gradient_boosting"] = {"dev": metrics(y_dev, hgb.predict(X_dev)),
                                         "train_seconds": round(time.time() - t, 1),
                                         "tuning": hgb_log}
    fitted["hist_gradient_boosting"] = hgb

    alpha = fitted["ridge_interactions"].named_steps["model"].regressor_.alpha_
    results["ridge_interactions"]["alpha"] = float(alpha)

    MODELS_DIR.mkdir(exist_ok=True)
    for name, model in fitted.items():
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        results[name]["model_size_mb"] = round(path.stat().st_size / 1e6, 2)

    best_overall = min(fitted, key=lambda k: results[k]["dev"]["MAE"])
    # The deployed model must fit in a GitHub repo / small cloud instance.
    deployable = [k for k in fitted if results[k]["model_size_mb"] <= MAX_DEPLOY_MB]
    best = min(deployable, key=lambda k: results[k]["dev"]["MAE"])
    print(f"Best on dev overall: {best_overall}; deployed (<= {MAX_DEPLOY_MB} MB): {best}")

    joblib.dump(fitted[best], MODELS_DIR / "fwi_model.joblib", compress=3)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "dev_results.json").write_text(json.dumps(
        {"best_model": best, "best_overall": best_overall, "results": results},
        indent=2, default=float))


if __name__ == "__main__":
    main()

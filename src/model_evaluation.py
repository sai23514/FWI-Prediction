"""Step 4 - evaluate every trained model on the held-out test set.

    python -m src.model_evaluation

Writes reports/test_results.json, reports/model_comparison.csv and plots
(actual vs predicted, residuals, error by FWI danger class, feature importance).
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR
from src.feature_engineering import split_xy
from src.fwi_equations import fwi_from_isi_bui
from src.model_development import metrics

MODEL_NAMES = ["linear_regression", "ridge_interactions", "random_forest",
               "hist_gradient_boosting"]
PRETTY = {"linear_regression": "Linear Regression (baseline)",
          "ridge_interactions": "Ridge + interactions",
          "random_forest": "Random Forest",
          "hist_gradient_boosting": "Gradient Boosting (HGB)",
          "fwi_equation": "Official FWI equation (reference)"}

# Canadian fire danger classes (CWFIS)
DANGER_BINS = [-0.1, 5, 10, 20, 30, np.inf]
DANGER_LABELS = ["Low (0-5)", "Moderate (5-10)", "High (10-20)", "Very high (20-30)",
                 "Extreme (30+)"]


def danger_class(fwi):
    return pd.cut(fwi, DANGER_BINS, labels=DANGER_LABELS)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    X, y = split_xy(test)

    preds = {n: np.clip(joblib.load(MODELS_DIR / f"{n}.joblib").predict(X), 0, None)
             for n in MODEL_NAMES}
    preds["fwi_equation"] = fwi_from_isi_bui(X["initial_spread_index"], X["build_up_index"])

    rows = []
    for n, p in preds.items():
        m = metrics(y, p)
        m["danger_class_accuracy"] = float((danger_class(p) == danger_class(y)).mean())
        rows.append({"model": PRETTY[n], **{k: round(v, 4) for k, v in m.items()}})
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(table.to_string(index=False))

    best = json.loads((REPORTS_DIR / "dev_results.json").read_text())["best_model"]
    p = preds[best]

    # MAE by danger class for the best model
    by_class = (pd.DataFrame({"cls": danger_class(y), "err": np.abs(y - p)})
                .groupby("cls", observed=False)["err"].agg(["count", "mean"])
                .rename(columns={"mean": "MAE"}).round(3))
    print(by_class)

    # --- plots ---
    rng = np.random.default_rng(RANDOM_STATE)
    s = rng.choice(len(y), size=min(20000, len(y)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, n in zip(axes, ["linear_regression", best]):
        ax.scatter(y.iloc[s], preds[n][s], s=3, alpha=0.3)
        lim = [0, max(y.max(), preds[n].max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set(title=f"{PRETTY[n]} - test set", xlabel="Actual FWI", ylabel="Predicted FWI")
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(p[s], (y.values - p)[s], s=3, alpha=0.3)
    ax.axhline(0, color="r", ls="--")
    ax.set(title=f"Residuals - {PRETTY[best]}", xlabel="Predicted FWI",
           ylabel="Actual - Predicted"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "residuals_best_model.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    tm = table[table.model != PRETTY["fwi_equation"]]
    ax.barh(tm.model, tm.MAE, color="#d9622b")
    for i, v in enumerate(tm.MAE):
        ax.text(v, i, f" {v:.3f}", va="center")
    ax.set(title="Test MAE by model (lower is better)", xlabel="MAE (FWI units)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "model_comparison_mae.png", dpi=120); plt.close(fig)

    model = joblib.load(MODELS_DIR / f"{best}.joblib")
    sub = rng.choice(len(y), size=5000, replace=False)
    imp = permutation_importance(model, X.iloc[sub], y.iloc[sub], n_repeats=3,
                                 random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
    imp_s = pd.Series(imp.importances_mean, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp_s.plot.barh(ax=ax, color="#2b6cd9")
    ax.set(title=f"Permutation importance - {PRETTY[best]}", xlabel="Increase in MAE")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=120); plt.close(fig)

    (REPORTS_DIR / "test_results.json").write_text(json.dumps({
        "best_model": best,
        "test_rows": int(len(y)),
        "comparison": rows,
        "best_model_mae_by_danger_class": by_class.reset_index().astype({"cls": str}).to_dict("records"),
        "permutation_importance": imp_s.sort_values(ascending=False).round(4).to_dict(),
    }, indent=2, default=float))


if __name__ == "__main__":
    main()

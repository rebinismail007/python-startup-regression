"""Reconstructed study exercise: a reproducible regression experiment on synthetic data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
TARGET = "Profit"

def validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if len(frame) < 20:
        raise ValueError("At least 20 rows are needed for this train/test experiment.")
    clean = frame.copy()
    for column in FEATURES + [TARGET]:
        clean[column] = pd.to_numeric(clean[column], errors="raise")
        if not np.isfinite(clean[column]).all():
            raise ValueError(f"{column} contains missing or non-finite values.")
    if (clean[FEATURES] < 0).any().any():
        raise ValueError("Spending cannot be negative. Zero spending is allowed.")
    return clean.reset_index(drop=True)

def train(frame: pd.DataFrame, seed: int = 42):
    data = validate(frame)
    train_rows, test_rows = train_test_split(np.arange(len(data)), test_size=0.2, random_state=seed)
    x_train, y_train = data.loc[train_rows, FEATURES], data.loc[train_rows, TARGET]
    x_test, y_test = data.loc[test_rows, FEATURES], data.loc[test_rows, TARGET]
    model = make_pipeline(StandardScaler(), LinearRegression())
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_rmse = -cross_val_score(model, x_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)
    scaler, linear = model.steps[0][1], model.steps[1][1]
    raw_coefficients = linear.coef_ / scaler.scale_
    raw_intercept = linear.intercept_ - float(np.dot(raw_coefficients, scaler.mean_))
    report = {
        "data_kind": "synthetic demonstration", "seed": seed,
        "rows": len(data), "train_rows": train_rows.tolist(), "test_rows": test_rows.tolist(),
        "features": FEATURES, "dropped_feature": "State",
        "cv_rmse_mean": float(cv_rmse.mean()), "cv_rmse_std": float(cv_rmse.std()),
        "holdout_rmse": float(root_mean_squared_error(y_test, predicted)),
        "holdout_mae": float(mean_absolute_error(y_test, predicted)),
        "holdout_r2": float(r2_score(y_test, predicted)),
        "baseline_rmse": float(root_mean_squared_error(y_test, baseline.predict(x_test))),
        "raw_intercept": float(raw_intercept),
        "raw_coefficients": dict(zip(FEATURES, map(float, raw_coefficients))),
        "holdout_actual": y_test.tolist(), "holdout_predicted": predicted.tolist(),
    }
    return model, report

def plot(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    actual = np.array(report["holdout_actual"])
    predicted = np.array(report["holdout_predicted"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].scatter(actual, predicted, color="#397d76")
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    axes[0].plot([low, high], [low, high], "--", color="#af563b")
    axes[0].set(xlabel="Actual synthetic profit", ylabel="Predicted synthetic profit", title="Untouched holdout set")
    axes[1].scatter(predicted, actual - predicted, color="#397d76")
    axes[1].axhline(0, linestyle="--", color="#af563b")
    axes[1].set(xlabel="Predicted synthetic profit", ylabel="Residual", title="What the model misses")
    for axis in axes:
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    fig.suptitle("Startup regression — synthetic learning data")
    fig.savefig(path, dpi=160)
    plt.close(fig)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "data/startups-demo.csv")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        _, report = train(pd.read_csv(args.csv), args.seed)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        plot(report, args.output / "holdout.png")
        print(f"Holdout RMSE: {report['holdout_rmse']:.2f} | Mean baseline: {report['baseline_rmse']:.2f}")
        print(f"Five-fold training CV RMSE: {report['cv_rmse_mean']:.2f}")
        print(f"Results written to {args.output}")
        return 0
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")

if __name__ == "__main__":
    raise SystemExit(main())

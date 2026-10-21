# Startup Regression — Python / pandas / scikit-learn

A reconstructed version of the study assignment about predicting startup profit from R&D, administration and marketing spending. **New code, October 2026.**

## Source and reconstruction

The supplied `Machinelearning Mandatory assignment` PDF describes the startup domain, dropping `State`, retaining genuine zero spending, training a linear regression and measuring RMSE. Page 13 includes a `pandas.read_csv` loading snippet. Most later code sections are placeholders rather than complete source.

This implementation completes the experiment using **80 newly generated synthetic rows**, not the original `50_Startups.csv`. The synthetic data has a deliberately learnable relationship; its scores do not establish performance on real companies.

## Run

Python 3.12+.

```sh
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python -m unittest discover -s tests -v
```

On Windows activate with `.venv\Scripts\activate`. The script writes `output/report.json` and a holdout/residual figure to `output/holdout.png`. Supply `--csv path.csv`, `--output folder` or `--seed 7` to change the experiment.

## What to inspect

- Validation of required numeric columns, finite values and nonnegative spending.
- A seeded 80/20 split before fitting any transformation.
- `StandardScaler → LinearRegression` pipeline; five-fold CV uses training rows only.
- Holdout RMSE, MAE, R² and a mean-prediction baseline.
- Conversion of standardized coefficients back to the original feature units.
- Four tests for validation, leakage prevention, a known synthetic relationship and reproducibility.

The PDF's request to predict investment risk is outside what a simple spending-to-profit regression demonstrates. This is a supervised regression exercise, with no risk rating or recommendation engine.

Primary references: [scikit-learn pipelines](https://scikit-learn.org/stable/common_pitfalls.html), [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

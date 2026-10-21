from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from train import FEATURES, TARGET, train, validate

class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(Path(__file__).parents[1] / "data/startups-demo.csv")

    def test_invalid_data_is_rejected_but_zero_spend_is_preserved(self):
        self.data.loc[0, FEATURES[0]] = 0
        self.assertEqual(validate(self.data).loc[0, FEATURES[0]], 0)
        self.data.loc[1, TARGET] = np.nan
        with self.assertRaises(ValueError): validate(self.data)
        with self.assertRaises(ValueError): validate(self.data.drop(columns=[TARGET]))

    def test_holdout_is_disjoint_and_scaler_only_sees_training_rows(self):
        model, report = train(self.data)
        self.assertFalse(set(report["train_rows"]) & set(report["test_rows"]))
        np.testing.assert_allclose(model.steps[0][1].mean_, self.data.loc[report["train_rows"], FEATURES].mean())
        self.assertEqual(len(report["test_rows"]) + len(report["train_rows"]), len(self.data))

    def test_model_beats_mean_baseline_on_known_synthetic_relationship(self):
        _, report = train(self.data)
        self.assertLess(report["holdout_rmse"], report["baseline_rmse"] * .3)
        self.assertGreater(report["holdout_r2"], .9)

    def test_reproducibility_and_raw_coefficient_conversion(self):
        model, first = train(self.data, 7)
        _, second = train(self.data, 7)
        self.assertEqual(first, second)
        row = self.data[FEATURES].iloc[[0]]
        manual = first["raw_intercept"] + sum(row.iloc[0][key] * value for key, value in first["raw_coefficients"].items())
        self.assertAlmostEqual(float(model.predict(row)[0]), manual, places=6)

if __name__ == "__main__": unittest.main()

import unittest

import numpy as np

from pls_sem import PLSSEMModel


class PLSSEMModelTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(13)
        predictor = random.normal(size=400)
        outcome = 0.8 * predictor + random.normal(scale=0.5, size=400)
        self.data = {
            "x1": predictor + random.normal(scale=0.15, size=400),
            "x2": predictor + random.normal(scale=0.15, size=400),
            "y1": outcome + random.normal(scale=0.15, size=400),
            "y2": outcome + random.normal(scale=0.15, size=400),
        }
        self.model = PLSSEMModel(
            {"Satisfaction": ["x1", "x2"], "Loyalty": ["y1", "y2"]},
            [("Satisfaction", "Loyalty")],
        )

    def test_fit_estimates_expected_relationship_and_measurement_results(self):
        fitted = self.model.fit(self.data)

        self.assertGreater(fitted.path_coefficients_[("Satisfaction", "Loyalty")], 0.7)
        self.assertGreater(fitted.r_squared_["Loyalty"], 0.5)
        self.assertEqual(set(fitted.loadings_), {"x1", "x2", "y1", "y2"})
        self.assertEqual(set(fitted.scores_), {"Satisfaction", "Loyalty"})
        self.assertGreater(fitted.n_iter_, 0)

    def test_mode_b_is_supported(self):
        model = PLSSEMModel(
            {"Satisfaction": ["x1", "x2"], "Loyalty": ["y1", "y2"]},
            [("Satisfaction", "Loyalty")],
            modes={"Satisfaction": "B"},
        ).fit(self.data)

        self.assertGreater(model.path_coefficients_[("Satisfaction", "Loyalty")], 0.7)

    def test_fit_rejects_missing_or_constant_indicators(self):
        with self.assertRaisesRegex(ValueError, "Missing indicator"):
            self.model.fit({"x1": self.data["x1"]})

        constant_data = {name: values.copy() for name, values in self.data.items()}
        constant_data["x1"] = np.ones(400)
        with self.assertRaisesRegex(ValueError, "no variation"):
            self.model.fit(constant_data)

    def test_fit_rejects_non_column_data(self):
        invalid_data = {name: values.copy() for name, values in self.data.items()}
        invalid_data["x1"] = np.column_stack((invalid_data["x1"], invalid_data["x1"]))

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            self.model.fit(invalid_data)

    def test_constructor_rejects_undefined_path(self):
        with self.assertRaisesRegex(ValueError, "defined constructs"):
            PLSSEMModel({"Satisfaction": ["x1"]}, [("Satisfaction", "Missing")])


if __name__ == "__main__":
    unittest.main()

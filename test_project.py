"""
============================================================
 NetSmart - Comprehensive Test Suite (Real Data)
 Run with:  python test_project.py
============================================================
"""

import unittest
import numpy as np
import pandas as pd

from config import (
    TIME_MAP, PURPOSE_MAP, DEVICE_MAP,
    FEATURE_COLS, SLOW_THRESHOLD,
    ONLINE_DATASET_PATH, FORMS_DATASET_PATH,
)
from data_loader   import loadOnlineDataset, loadFormsDataset, loadCombinedDataset
from model_trainer import encodeFeatures, trainAllModels, predictSpeed, compareDatasets
from prob_stats_tools import (
    gaussianSlowProbability,
    poissonInstantProbability,
    binomialWeeklyReliability,
)


class TestNetSmart(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load real datasets and train models once."""
        cls.df_online   = loadOnlineDataset()
        cls.df_forms    = loadFormsDataset()
        cls.df_combined = loadCombinedDataset()
        cls.lr, cls.br, cls.gb, cls.gbUL, cls.metrics, cls.pModel, cls.splitInfo = trainAllModels(cls.df_combined)

    # ── Data Loading Tests ────────────────────────────────────────

    def test_online_dataset_loads(self):
        self.assertGreater(len(self.df_online), 100)
        for col in ["TimeOfDay", "Purpose", "Device", "SpeedDownload", "IsSlow"]:
            self.assertIn(col, self.df_online.columns)

    def test_forms_dataset_loads(self):
        self.assertGreater(len(self.df_forms), 50)
        for col in ["TimeOfDay", "Purpose", "Device", "SpeedDownload", "IsSlow"]:
            self.assertIn(col, self.df_forms.columns)

    def test_combined_is_union(self):
        self.assertEqual(len(self.df_combined), len(self.df_online) + len(self.df_forms))

    def test_slow_flag_logic(self):
        for _, row in self.df_combined.head(30).iterrows():
            expected = 1 if row["SpeedDownload"] < SLOW_THRESHOLD else 0
            self.assertEqual(row["IsSlow"], expected)

    # ── Encoding Tests ────────────────────────────────────────────

    def test_encoding(self):
        enc = encodeFeatures(self.df_combined.head(10))
        self.assertIn("TimeEncoded", enc.columns)
        self.assertIn("PurposeEncoded", enc.columns)
        self.assertIn("DeviceEncoded", enc.columns)

    # ── Model Tests ───────────────────────────────────────────────

    def test_all_four_models_present(self):
        for m in ["Linear Regression", "Bayesian Model", "Gradient Boosting", "Parametric Stats"]:
            self.assertIn(m, self.metrics)

    def test_split_is_80_10_10(self):
        total = self.splitInfo["total"]
        self.assertAlmostEqual(self.splitInfo["train"] / total, 0.8, delta=0.05)
        self.assertAlmostEqual(self.splitInfo["validation"] / total, 0.1, delta=0.05)
        self.assertAlmostEqual(self.splitInfo["test"] / total, 0.1, delta=0.05)

    def test_prediction_output(self):
        lrS, brS, gbS, ulS, pS, isS, prob = predictSpeed(
            "Evening", "Gaming", "Mobile", 300, 1,
            self.lr, self.br, self.gb, self.gbUL, self.pModel,
        )
        self.assertGreaterEqual(gbS, 0)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    # ── Probability Tests ─────────────────────────────────────────

    def test_gaussian(self):
        p = gaussianSlowProbability(mu=SLOW_THRESHOLD, sigma=2.0)
        self.assertAlmostEqual(p, 0.5)

    def test_poisson(self):
        p10 = poissonInstantProbability(k=10, lam=10.0)
        p20 = poissonInstantProbability(k=10, lam=30.0)
        self.assertGreater(p10, p20)

    def test_binomial(self):
        prob = binomialWeeklyReliability(p_slow=0.5, n_days=7, min_slow_days=3)
        self.assertGreater(prob, 0.5)

    # ── Comparison Tests ──────────────────────────────────────────

    def test_ttest(self):
        z1, z2, t, p = compareDatasets(self.df_online, self.df_forms)
        self.assertEqual(len(z1), len(self.df_online))
        self.assertEqual(len(z2), len(self.df_forms))
        self.assertIsInstance(t, float)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" RUNNING NETSMART REAL-DATA TEST SUITE")
    print("=" * 50 + "\n")
    unittest.main()

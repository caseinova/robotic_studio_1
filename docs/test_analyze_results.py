"""Offline checks for the published metric definitions."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_results import AXES, analyze, metrics


class ResultsTests(unittest.TestCase):
    def test_rmse_is_vector_norm_not_mean_axis_rmse(self):
        errors = np.array([[.003, .004, 0, 0, 0, np.pi / 2],
                           [0, 0, 0, 0, 0, 0]])
        result = metrics(errors)
        self.assertAlmostEqual(result["position_rmse_mm"], 5 / np.sqrt(2))
        self.assertAlmostEqual(result["rotation_rmse_deg"], 90 / np.sqrt(2))
        self.assertIsNone(metrics(np.empty((0, 6))))

    def test_latched_and_missing_samples_are_not_live_errors(self):
        rows = []
        for index, status in enumerate(("raw_cube_tag", "latched_pick_target", "unavailable")):
            row = {"time_s": index * .05}
            for obj in ("cube", "tool"):
                row[obj + "_status"] = status if obj == "cube" else "fk"
                errors = ([.003, .004, 0, 0, 0, 0] if index == 0 else
                          [.1, 0, 0, 0, 0, 0] if index == 1 else [float("nan")] * 6)
                for axis, value in zip(AXES, errors):
                    row[f"{obj}_error_{axis}"] = value
                    row[f"{obj}_std_{axis}"] = .01
                for i, axis in enumerate("xyz"):
                    row[f"{obj}_true_{axis}_m"] = 0
                    row[f"{obj}_estimate_{axis}_m"] = errors[i]
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "estimation.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            result = analyze(path)
        cube = result["cube"]
        self.assertEqual(cube["live_samples"], 1)
        self.assertEqual(cube["finite_samples"], 2)
        self.assertAlmostEqual(cube["live_metrics"]["position_rmse_mm"], 5)
        self.assertAlmostEqual(cube["by_status"]["latched_pick_target"]["position_rmse_mm"], 100)
        self.assertIsNone(cube["by_status"]["unavailable"])
        self.assertEqual(cube["uncertainty"]["samples"], 1)
        self.assertEqual(cube["uncertainty"]["marginal_2sigma_coverage_xyz_rxyz"], [1] * 6)


if __name__ == "__main__":
    unittest.main()

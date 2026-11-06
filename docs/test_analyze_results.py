"""Offline checks for the published metric definitions."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_results import AXES, ROOT, analyze, destination_metrics, metrics


class ResultsTests(unittest.TestCase):
    def test_archived_comparison_settings_match_reported_design(self):
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        runs = {run['label']: run for run in manifest['runs']}
        self.assertEqual(set(runs), {f'{group}{i}' for group in 'LHO' for i in range(1, 5)})

        def config(label, name):
            return json.loads((ROOT / runs[label]['id'] / (name + '_config.json')).read_text())

        for i in range(1, 5):
            for name in ('controller', 'estimator', 'sequence', 'simulation'):
                self.assertEqual(config(f'L{i}', name), config(f'H{i}', name))
            low, high = config(f'L{i}', 'noise'), config(f'H{i}', 'noise')
            for key in ('joint_position_std', 'joint_velocity_std', 'depth_std', 'depth_std_per_meter_squared'):
                self.assertAlmostEqual(high[key], 3 * low[key])
            self.assertEqual(high['rgb_pixel_std'], 10 * low['rgb_pixel_std'])
            for name in ('controller', 'noise', 'sequence', 'simulation'):
                self.assertEqual(config(f'H{i}', name), config(f'O{i}', name))
            filtered, unfiltered = config(f'H{i}', 'estimator'), config(f'O{i}', 'estimator')
            for key in ('cube_enabled', 'tool_enabled'):
                self.assertTrue(filtered.pop(key))
                self.assertFalse(unfiltered.pop(key))
            self.assertEqual(filtered, unfiltered)
            self.assertEqual(config(f'O{i}', 'noise')['depth_std_per_meter_squared'], .0015)

    def test_selection_excludes_video_and_superseded_trials(self):
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        report = json.loads((ROOT / 'summary.json').read_text())
        selected = {run['id'] for run in manifest['runs']}
        self.assertEqual(selected, {path.name for path in ROOT.iterdir() if path.is_dir()})
        self.assertEqual(selected, {run['id'] for run in report['runs']})
        excluded = {'new_dest_1', 'new_dest_2', '20261106-223439-500806-5bb5ad1b',
                    '20261106-225013-504042-a1b158f3',
                    '20261106-225133-751042-d2e90777', '20261106-225333-811285-db2d2751',
                    '20261106-225514-987711-28f3b95b', '20261106-225727-355524-bbc4cc7c'}
        self.assertTrue(excluded <= {run['id'] for run in manifest['excluded']})
        self.assertFalse(selected & excluded)
        page = (ROOT.parent / 'index.html').read_text()
        for run_id in selected:
            self.assertIn(f'id="result-{run_id}"', page)
        for run_id in excluded:
            self.assertNotIn(f'results/{run_id}/', page)
            self.assertNotIn(f'id="result-{run_id}"', page)
        for run in report['runs']:
            if run['group'] == 'high_noise_no_ekf':
                self.assertNotIn('uncertainty', run['cube'])
                self.assertNotIn('uncertainty', run['tool'])

    def test_published_table_and_summary_match_archived_csvs(self):
        report = json.loads((ROOT / 'summary.json').read_text())
        page = (ROOT.parent / 'index.html').read_text()
        for run in report['runs']:
            actual = analyze(ROOT / run['id'] / 'estimation.csv')
            for key, value in actual.items():
                self.assertEqual(run[key], value, (run['label'], key))
            cube, tool = run['cube'], run['tool']
            row = (f'<tr><th scope="row"><a href="#result-{run["id"]}">{run["label"]}</a></th>'
                   f'<td>{run["duration_s"]:.2f}</td>'
                   f'<td>{cube["live_first_s"]:.2f}\u2013{cube["live_last_s"]:.2f}</td>'
                   f'<td>{cube["live_samples"]} / {run["samples"]}</td>'
                   f'<td>{cube["live_metrics"]["position_rmse_mm"]:.3f} / '
                   f'{cube["live_metrics"]["rotation_rmse_deg"]:.3f}</td>'
                   f'<td>{tool["live_metrics"]["position_rmse_mm"]:.3f} / '
                   f'{tool["live_metrics"]["rotation_rmse_deg"]:.3f}</td></tr>')
            self.assertIn(row, page)

    def test_endpoint_metric_uses_xy_not_release_height(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'simulation_config.json').write_text(json.dumps({
                'destination_positions': {'drop': [.12, .32, 0]}}))
            (root / 'request.json').write_text(json.dumps({'destination': 'drop'}))
            result = destination_metrics(root, [.123, .324, .032])
        self.assertAlmostEqual(result['endpoint_xy_distance_to_marker_mm'], 5)
        self.assertEqual(result['configured_destination_marker_xyz_m'], [.12, .32, 0])

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

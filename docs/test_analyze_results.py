"""Offline checks for the published metric definitions."""
import csv
import json
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from analyze_results import AXES, ROOT, analyze, destination_metrics, metrics


class ResultsTests(unittest.TestCase):
    def test_frame_diagram_and_authorship_declaration(self):
        page = (ROOT.parent / 'index.html').read_text()
        diagram = page.split('id="frame-diagram"', 1)[1].split('</figure>', 1)[0]
        self.assertIn('assets/ur3_transform_tree.svg', diagram)
        svg = ET.parse(ROOT.parent / 'assets' / 'ur3_transform_tree.svg').getroot()
        diagram_text = ' '.join(svg.itertext())
        for label in ('world_from_base', 'world_from_camera', 'base_from_flange(q)',
                      'flange_from_tool', 'tool_from_marker', 'tool_from_grasp',
                      'inverse(cube_from_marker_i)', 'marker_from_cube'):
            self.assertIn(label, diagram_text)
        self.assertNotIn('UR3/UR5', diagram_text)
        self.assertNotIn('TCP / gripper', diagram_text)
        self.assertIn('parent_from_child maps child coordinates into the parent', diagram_text)
        paths = svg.findall('{http://www.w3.org/2000/svg}path')
        observations = [path for path in paths if path.get('stroke-dasharray')]
        self.assertEqual(len(observations), 3)
        self.assertTrue(all(path.get('marker-end') == 'url(#arrowo)' for path in observations))
        self.assertIn('id="authorship-table"', page)
        self.assertIn('[Full name]', page)
        self.assertIn('recommended by my tutor', page)
        self.assertIn('drafting and editing code', page)
        self.assertNotIn('Tutor-directed camera-to-hand work', page)

    def test_documented_local_rank_analysis(self):
        dt = .05
        controllability = np.hstack([dt * np.eye(6)] * 6)
        observability = np.vstack([np.eye(6)] * 6)
        self.assertEqual(np.linalg.matrix_rank(controllability), 6)
        self.assertEqual(np.linalg.matrix_rank(observability), 6)
        np.testing.assert_allclose(np.linalg.svd(controllability, compute_uv=False),
                                   np.sqrt(6) * dt)
        np.testing.assert_allclose(np.linalg.svd(observability, compute_uv=False), np.sqrt(6))
        for step, rank in ((dt, 12), (0., 6)):
            transition = np.eye(12)
            transition[:6, 6:] = step * np.eye(6)
            measurement = np.eye(12)[:6]
            self.assertEqual(np.linalg.matrix_rank(np.vstack(
                [measurement, measurement @ transition])), rank)

    def test_published_transfer_samples_match_diagnostics(self):
        log = (ROOT / '20261106-221048-444601-532fb743' / 'diagnostics.txt').read_text()
        page = (ROOT.parent / 'index.html').read_text()
        table = page.split('id="transfer-response"', 1)[1].split('</table>', 1)[0]
        rows = re.findall(r'<tr><td>([\d.]+)</td><td>([\d.]+)</td>'
                          r'<td>([\d.]+)</td><td>([\d.]+)</td></tr>', table)
        self.assertEqual(len(rows), 3)
        for elapsed, position, rotation, command in rows:
            samples = [line for line in log.splitlines()
                       if f'phase=transferring, tool=tag_fused, elapsed_s={elapsed},' in line]
            self.assertEqual(len(samples), 1)
            for field, value in (('position_error_mm', position),
                                 ('rotation_error_deg', rotation),
                                 ('commanded_max_joint_speed_rad_s', command)):
                self.assertIn(f'{field}={value},', samples[0])

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
        self.assertEqual(selected | {'hardware_results'},
                         {path.name for path in ROOT.iterdir() if path.is_dir()})
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

    def test_hardware_evidence_assets_are_linked_and_separate(self):
        page = (ROOT.parent / 'index.html').read_text()
        self.assertLess(page.index('<h2>Simulation results</h2>'),
                        page.index('id="result-20261106-232021-602378-0843680e"'))
        self.assertLess(page.index('id="result-20261106-232021-602378-0843680e"'),
                        page.index('<h2 id="hardware-results">Hardware pick-and-place results</h2>'))
        calibration = ROOT / 'hardware_results' / 'calibration_run'
        images = sorted(calibration.glob('calibration_[0-9][0-9]_*.png'))
        self.assertEqual(len(images), 19)
        self.assertFalse(list((ROOT / 'hardware_results').glob('Screenshot*.png')))
        for index, path in enumerate(images, 1):
            self.assertTrue(path.name.startswith(f'calibration_{index:02d}_'))
        assets = ['calibration_run/calibration_positions.png',
                  'calibration_run/calibration_positions.html']
        assets += [f'hardware_runs/{phase}{i}.png'
                   for i in range(1, 4) for phase in ('pick', 'place')]
        for asset in assets:
            self.assertTrue((ROOT / 'hardware_results' / asset).is_file())
            self.assertIn(f'results/hardware_results/{asset}', page)

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

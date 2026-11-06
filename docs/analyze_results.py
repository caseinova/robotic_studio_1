"""Reproduce the public results from archived CSVs, without running the robot.

Run: python docs/analyze_results.py
Dependencies: NumPy and Matplotlib from the project requirements.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "results"
AXES = ("x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad")


def metrics(errors):
    if not len(errors):
        return None
    position = np.linalg.norm(errors[:, :3], axis=1) * 1000
    rotation = np.rad2deg(np.linalg.norm(errors[:, 3:], axis=1))
    return {
        "samples": len(errors),
        "position_rmse_mm": float(np.sqrt(np.mean(position**2))),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rotation**2))),
        "position_p95_mm": float(np.percentile(position, 95)),
        "position_max_mm": float(position.max()),
        "mean_position_error_mm": (errors[:, :3].mean(axis=0) * 1000).tolist(),
    }


def analyze(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([float(row["time_s"]) for row in rows])
    assert len(rows) and np.all(np.diff(times) >= 0), path
    result = {
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "samples": len(rows),
        "duration_s": float(times[-1] - times[0]),
        "cube_start_xyz_m": [float(rows[0][f"cube_true_{a}_m"]) for a in "xyz"],
        "cube_end_xyz_m": [float(rows[-1][f"cube_true_{a}_m"]) for a in "xyz"],
    }
    for obj in ("cube", "tool"):
        errors = np.array([[float(r[f"{obj}_error_{a}"]) for a in AXES] for r in rows])
        sigma = np.array([[float(r[f"{obj}_std_{a}"]) for a in AXES] for r in rows])
        statuses = np.array([r[f"{obj}_status"] for r in rows])
        valid = np.isfinite(errors).all(axis=1) & ~np.isin(statuses, ["unavailable", "invalid"])
        # A latched target is a fixed reference, not a moving-cube estimate.
        live = valid & (statuses != "latched_pick_target")
        recorded = {
            "status_counts": dict(Counter(statuses.tolist())),
            "finite_samples": int(valid.sum()),
            "live_samples": int(live.sum()),
            "live_recording_fraction": float(live.mean()),
            "live_first_s": float(times[live][0] - times[0]) if live.any() else None,
            "live_last_s": float(times[live][-1] - times[0]) if live.any() else None,
            "live_metrics": metrics(errors[live]),
            "by_status": {status: metrics(errors[valid & (statuses == status)])
                          for status in sorted(set(statuses))},
        }
        # Check that the exported signed translation errors match the poses.
        truth = np.array([[float(r[f"{obj}_true_{a}_m"]) for a in "xyz"] for r in rows])
        estimate = np.array([[float(r[f"{obj}_estimate_{a}_m"]) for a in "xyz"] for r in rows])
        assert np.allclose(estimate[valid] - truth[valid], errors[valid, :3], atol=1e-10), path
        with_covariance = live & np.isfinite(sigma).all(axis=1) & (sigma >= 0).all(axis=1)
        if with_covariance.any():
            e, std = errors[with_covariance], sigma[with_covariance]
            recorded["uncertainty"] = {
                "samples": int(with_covariance.sum()),
                "marginal_2sigma_coverage_xyz_rxyz": (np.abs(e) <= 2 * std).mean(axis=0).tolist(),
                "median_position_sigma_mm": float(np.median(std[:, :3]) * 1000),
                "median_rotation_sigma_deg": float(np.rad2deg(np.median(std[:, 3:]))),
                "position_axis_rmse_mm": (np.sqrt(np.mean(e[:, :3]**2, axis=0)) * 1000).tolist(),
                "rotation_axis_rmse_deg": np.rad2deg(np.sqrt(np.mean(e[:, 3:]**2, axis=0))).tolist(),
            }
        result[obj] = recorded
    return result


def plot(runs):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(13, 8), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    groups = [('low_noise', 'Low noise, EKFs on', '#197b67'),
              ('high_noise', 'High noise, EKFs on', '#496b94'),
              ('high_noise_no_ekf', 'High noise, EKFs off', '#a34337')]
    layouts = sorted({r['layout'] for r in runs})
    positions = np.arange(len(layouts))
    lookup = {(r['group'], r['layout']): r for r in runs}
    for row, obj in enumerate(("cube", "tool")):
        for col, (metric, unit) in enumerate((("position_rmse_mm", "mm"), ("rotation_rmse_deg", "deg"))):
            ax = axes[row, col]
            maximum = 0
            for index, (group, label, color) in enumerate(groups):
                values = [lookup[group, layout][obj]['live_metrics'][metric] for layout in layouts]
                bars = ax.bar(positions + (index - 1) * .25, values, color=color, width=.24, label=label)
                ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
                maximum = max(maximum, max(values))
            ax.set_xticks(positions, [f'Position {layout}' for layout in layouts])
            ax.set_ylim(0, maximum * 1.25)
            ax.set_title(f"{obj.title()} {'position' if col == 0 else 'orientation'} RMSE")
            ax.set_ylabel(unit)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Recorded pose error at four matched initial cube positions\nHigh-noise EKF-on/off pairs use identical noise settings", fontsize=12)
    fig.savefig(ROOT / "summary.png", dpi=150)


def destination_metrics(directory, endpoint):
    """Horizontal endpoint-to-marker distance, not full release-pose error."""
    scene = json.loads((directory / 'simulation_config.json').read_text())
    request = json.loads((directory / 'request.json').read_text())
    marker = scene['destination_positions'][request['destination']]
    return {
        'configured_destination_marker_xyz_m': marker,
        'endpoint_xy_distance_to_marker_mm': float(
            np.linalg.norm(np.asarray(endpoint)[:2] - np.asarray(marker)[:2]) * 1000),
    }


def plot_trajectories(runs):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 6), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    for ax, labels, title in zip(axes,
            [('L1', 'L2', 'L3', 'L4', 'H1', 'H2', 'H3', 'H4'),
             ('H1', 'H2', 'H3', 'H4', 'O1', 'O2', 'O3', 'O4')],
            ['Low versus high noise, EKFs on', 'High noise, EKFs on versus off']):
        markers = set()
        for run in runs:
            if run['label'] not in labels:
                continue
            with (ROOT / run['id'] / 'estimation.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            xy = np.array([[float(row[f'cube_true_{a}_m']) * 1000 for a in 'xy'] for row in rows])
            style = '--' if run['group'] == 'high_noise' else '-'
            ax.plot(*xy.T, style, linewidth=1.7, label=run['label'])
            ax.scatter(*xy[0], marker='o', s=20, color='black', zorder=4)
            ax.scatter(*xy[-1], marker='x', s=40, color='black', zorder=4)
            marker = run.get('configured_destination_marker_xyz_m')
            if marker is not None:
                markers.add(tuple(marker[:2]))
        for xy in sorted(markers):
            ax.scatter(*(np.array(xy)*1000), marker='*', s=180, facecolor='none',
                       edgecolor='black', zorder=5)
        ax.set(title=title, xlabel='World X (mm)', ylabel='World Y (mm)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.25)
        ax.legend(ncol=2, fontsize=9)
    fig.suptitle('True cube trajectories: same starts and destination\nCircles: starts | crosses: endpoints | stars: goal', fontsize=12)
    fig.savefig(ROOT / 'trajectories.png', dpi=150)


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    runs = [{**run, **analyze(ROOT / run["id"] / "estimation.csv")} for run in manifest["runs"]]
    for run in runs:
        if run['configuration_available']:
            run.update(destination_metrics(ROOT / run['id'], run['cube_end_xyz_m']))
    report = {
        "method": "Sample-weighted RMS of 3D translation-error norm and rotation-vector norm. Finite live rows only; latched targets excluded. No interpolation or pooling across runs. Coverage is marginal per axis on finite covariance rows, not joint coverage or a safety bound. Durations are simulation-time recording spans, not wall-clock completion times. Recording fractions are not detector success rates.",
        "runs": runs,
        "excluded": manifest.get('excluded', []),
    }
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot(runs)
    plot_trajectories(runs)
    print("Wrote results/summary.json, summary.png and trajectories.png for", len(runs), "runs")


if __name__ == "__main__":
    main()

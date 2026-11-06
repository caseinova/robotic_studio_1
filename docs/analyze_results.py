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

    fig = Figure(figsize=(11, 7), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    labels = [r["label"] for r in runs]
    colors = ["#197b67" if r["id"] in ("1", "2", "3", "4") else
              "#a34337" if r["id"] == "without_ekf" else "#496b94" for r in runs]
    for row, obj in enumerate(("cube", "tool")):
        for col, (metric, unit) in enumerate((("position_rmse_mm", "mm"), ("rotation_rmse_deg", "deg"))):
            ax = axes[row, col]
            values = [r[obj]["live_metrics"][metric] for r in runs]
            bars = ax.bar(labels, values, color=colors, width=.65)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
            ax.set_ylim(0, max(values) * 1.25)
            ax.set_title(f"{obj.title()} {'position' if col == 0 else 'orientation'} RMSE")
            ax.set_ylabel(unit)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    fig.suptitle("Recorded pose error against simulator truth\nDifferent runs and observation windows; not a paired EKF benchmark", fontsize=13)
    fig.savefig(ROOT / "summary.png", dpi=150)


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    runs = [{**run, **analyze(ROOT / run["id"] / "estimation.csv")} for run in manifest["runs"]]
    report = {
        "method": "Sample-weighted RMS of 3D translation-error norm and rotation-vector norm. Finite live rows only; latched targets excluded. No interpolation or pooling across runs. Coverage is marginal per axis on finite covariance rows, not joint coverage or a safety bound. Durations are simulation-time recording spans, not wall-clock completion times. Recording fractions are not detector success rates.",
        "runs": runs,
    }
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot(runs)
    print("Wrote results/summary.json and results/summary.png for", len(runs), "runs")


if __name__ == "__main__":
    main()

"""Simulation-only truth/estimator reporting; never used for control."""
import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _truth(data, kind):
    obj = data.body('cube') if kind == 'cube' else data.site('attachment_site')
    pose = np.eye(4)
    pose[:3, 3] = obj.xpos
    pose[:3, :3] = obj.xmat.reshape(3, 3)
    return pose


class EstimationTelemetry:
    def __init__(self):
        self.samples = []
        self.latest = {}

    def record(self, data, *, cube_estimate=None, cube_pose=None, tool_pose=None, tool_covariance=None,
               tool_status='unavailable', cube_status='unavailable'):
        sample = {'time': float(data.time)}
        self.latest = {}
        for kind, estimate, covariance, status in (
            ('cube', cube_pose if cube_estimate is None else cube_estimate.world_from_cube,
             None if cube_estimate is None else cube_estimate.covariance, cube_status),
            ('tool', tool_pose, tool_covariance, tool_status),
        ):
            truth = _truth(data, kind)
            sample[kind + '_true_quaternion_xyzw'] = Rotation.from_matrix(truth[:3, :3]).as_quat()
            sample[kind + '_estimate_quaternion_xyzw'] = np.full(4, np.nan)
            valid = estimate is not None
            error = np.full(6, np.nan)
            sigma = np.full(6, np.nan)
            if valid:
                estimate = np.asarray(estimate, dtype=float).copy()
                if covariance is not None:
                    covariance = np.asarray(covariance, dtype=float)[:6, :6].copy()
                if (estimate.shape == (4, 4) and np.all(np.isfinite(estimate))
                        and (covariance is None or (covariance.shape == (6, 6) and np.all(np.isfinite(covariance))))):
                    error[:3] = estimate[:3, 3] - truth[:3, 3]
                    error[3:] = Rotation.from_matrix(estimate[:3, :3] @ truth[:3, :3].T).as_rotvec()
                    if covariance is not None:
                        sigma = np.sqrt(np.maximum(np.diag(covariance), 0))
                    sample[kind + '_estimate_quaternion_xyzw'] = Rotation.from_matrix(estimate[:3, :3]).as_quat()
                    self.latest[kind] = (estimate, covariance)
                else:
                    status = 'invalid'
            sample[kind] = (truth[:3, 3].copy(), error, sigma, status)
        self.samples.append(sample)

    def save(self, directory=None):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        if directory is None:
            directory = Path(__file__).resolve().parents[1] / 'simulation_plots' / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + uuid4().hex[:8])
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        figure = Figure(figsize=(13, 11), constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.subplots(3, 2, sharex=True)
        if self.samples:
            times = np.array([s['time'] for s in self.samples])
            times -= times[0]
            for column, kind in enumerate(('cube', 'tool')):
                truth = np.array([s[kind][0] for s in self.samples])
                error = np.array([s[kind][1] for s in self.samples])
                sigma = np.array([s[kind][2] for s in self.samples])
                statuses = np.array([s[kind][3] for s in self.samples])
                for i, ax in enumerate(axes[:, column]):
                    ax.plot(times, truth[:, i], label='True', color='#203e58')
                    for index, status in enumerate(sorted(set(statuses) - {'unavailable', 'invalid'})):
                        values = np.where(statuses == status, truth[:, i] + error[:, i], np.nan)
                        ax.plot(times, values, '--', marker='.', markersize=3,
                                color=('#d55e00', '#009e73', '#8e44ad', '#b58c00')[index % 4],
                                label=status.replace('_', ' '))
                    if np.any(np.isfinite(sigma[:, i])):
                        ax.fill_between(times, truth[:, i] + error[:, i] - 2*sigma[:, i],
                                        truth[:, i] + error[:, i] + 2*sigma[:, i], color='#d55e00', alpha=.18,
                                        label='EKF estimate ±2σ')
                    ax.set_ylabel('XYZ'[i] + ' (m)')
                    ax.grid(alpha=.25)
                axes[0, column].set_title(kind.title() + ' position in world')
                axes[0, column].legend(fontsize=8)
                axes[-1, column].set_xlabel('Simulation time (s)')
        else:
            axes[1, 0].text(.5, .5, 'No estimator samples', ha='center', transform=axes[1, 0].transAxes)
        figure.suptitle('True and estimated positions (shading: filter ±2σ, not a safety bound)')
        image = directory / 'estimation.png'
        figure.savefig(image, dpi=140)
        orientation = Figure(figsize=(13, 11), constrained_layout=True)
        FigureCanvasAgg(orientation)
        orient_axes = orientation.subplots(3, 2, sharex=True)
        if self.samples:
            for column, kind in enumerate(('cube', 'tool')):
                error = np.array([s[kind][1] for s in self.samples])
                sigma = np.array([s[kind][2] for s in self.samples])
                statuses = np.array([s[kind][3] for s in self.samples])
                for i, ax in enumerate(orient_axes[:, column]):
                    ax.axhline(0, color='#203e58', label='True')
                    for index, status in enumerate(sorted(set(statuses) - {'unavailable', 'invalid'})):
                        ax.plot(times, np.where(statuses == status, error[:, i+3], np.nan),
                                '--', marker='.', markersize=3,
                                color=('#d55e00', '#009e73', '#8e44ad', '#b58c00')[index % 4],
                                label=status.replace('_', ' '))
                    if np.any(np.isfinite(sigma[:, i+3])):
                        ax.fill_between(times, error[:, i+3] - 2*sigma[:, i+3],
                                        error[:, i+3] + 2*sigma[:, i+3], color='#d55e00', alpha=.18,
                                        label='EKF estimate ±2σ')
                    ax.set_ylabel('XYZ'[i] + ' rotation error (rad)')
                    ax.grid(alpha=.25)
                orient_axes[0, column].set_title(kind.title() + ' world-axis orientation error')
                orient_axes[0, column].legend(fontsize=8)
                orient_axes[-1, column].set_xlabel('Simulation time (s)')
        else:
            orient_axes[1, 0].text(.5, .5, 'No estimator samples', ha='center', transform=orient_axes[1, 0].transAxes)
        orientation.suptitle('Orientation relative to true pose (world-axis rotation vector; ±2σ is local linear approximation)')
        orientation.savefig(directory / 'orientation_error.png', dpi=140)
        csv_path = directory / 'estimation.csv'
        headers = ['time_s']
        for kind in ('cube', 'tool'):
            headers += [kind + '_status'] + [f'{kind}_true_{x}_m' for x in 'xyz']
            headers += [f'{kind}_estimate_{x}_m' for x in 'xyz']
            headers += [f'{kind}_{which}_quaternion_{x}' for which in ('true', 'estimate') for x in 'xyzw']
            headers += [f'{kind}_error_{x}_{unit}' for x, unit in zip(('x','y','z','rx','ry','rz'), ('m','m','m','rad','rad','rad'))]
            headers += [f'{kind}_std_{x}_{unit}' for x, unit in zip(('x','y','z','rx','ry','rz'), ('m','m','m','rad','rad','rad'))]
        with csv_path.open('w', newline='', encoding='utf-8') as output:
            writer = csv.writer(output)
            writer.writerow(headers)
            for sample in self.samples:
                row = [sample['time']]
                for kind in ('cube', 'tool'):
                    truth, error, sigma, status = sample[kind]
                    row.extend([status, *truth, *(truth + error[:3]),
                                *sample[kind + '_true_quaternion_xyzw'],
                                *sample[kind + '_estimate_quaternion_xyzw'], *error, *sigma])
                writer.writerow(row)
        return image, csv_path

    def draw(self, viewer, *, uncertainty_scale=1.0):
        """Draw pose axes and display-scaled ellipsoids without changing covariance."""
        if (isinstance(uncertainty_scale, bool)
                or not isinstance(uncertainty_scale, (int, float))
                or not np.isfinite(uncertainty_scale) or uncertainty_scale <= 0):
            raise ValueError('uncertainty_scale must be finite and positive')
        label = f'2 sigma x {uncertainty_scale:g} (display only)'
        with viewer.lock():
            scene = viewer.user_scn
            scene.ngeom = 0
            for kind, rgba in (('cube', [.1, .85, .35, .24]), ('tool', [1., .45, .1, .24])):
                if kind not in self.latest:
                    continue
                pose, covariance = self.latest[kind]
                if scene.ngeom + (6 if covariance is not None else 4) > scene.maxgeom:
                    continue
                if covariance is not None:
                    values, vectors = np.linalg.eigh((covariance[:3, :3] + covariance[:3, :3].T) / 2)
                    if np.linalg.det(vectors) < 0:
                        vectors[:, 0] *= -1
                    radii = np.maximum(uncertainty_scale*2*np.sqrt(np.maximum(values, 0)), 1e-5)
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        radii, pose[:3, 3], vectors.ravel(), np.array(rgba, dtype=np.float32))
                    scene.ngeom += 1
                    # Anchor text above the rotated ellipsoid's world-Z extent.
                    label_position = pose[:3, 3].copy()
                    label_position[2] += np.linalg.norm(vectors[2, :] * radii) + .012
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                        label_position, np.eye(3).ravel(), np.ones(4, dtype=np.float32))
                    geom.label = f'{kind}: {label}'
                    scene.ngeom += 1
                for axis, color in enumerate(([.95, .2, .2, 1.], [.15, .9, .25, 1.], [.2, .45, 1., 1.])):
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                        np.zeros(3), np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .002,
                        pose[:3, 3], pose[:3, 3] + .05 * pose[:3, axis])
                    scene.ngeom += 1
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.006, 0, 0]),
                    pose[:3, 3], np.eye(3).ravel(), np.array(rgba[:3] + [1.], dtype=np.float32))
                scene.ngeom += 1

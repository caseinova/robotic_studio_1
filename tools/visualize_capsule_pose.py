"""Render the recorded pickup failure using configured capsules, without hardware."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/robotic-studio-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hardware_config import HardwareConfig

FAILING_Q = [-1.087698761616842, -1.5815416775145472, -1.6191456317901611,
             -1.5562566083720704, 1.6523456573486328, -.3832309881793421]
COLORS = ['#63727a', '#c49a35', '#16a39a', '#547fa8', '#d84e55', '#8b77ac', '#eb963c']
LABELS = ['Base', 'Shoulder', 'Upper arm', 'Forearm', 'Wrist 1', 'Wrist 2', 'Wrist 3']


def surface(a, b, radius):
    length = np.linalg.norm(b-a)
    axis = (b-a)/length if length > 1e-12 else np.array([0., 0., 1.])
    seed = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angle = np.linspace(0, 2*np.pi, 32)
    t = np.linspace(-np.pi/2, np.pi/2, 24)
    z = np.concatenate((radius*np.sin(t[:12]), [0., length], length+radius*np.sin(t[12:])))
    radial = np.sqrt(np.maximum(0, radius**2 - np.maximum(-z, np.maximum(z-length, 0))**2))
    return (a + z[:, None, None]*axis + radial[:, None, None] *
            (np.cos(angle)[None, :, None]*u + np.sin(angle)[None, :, None]*v))*1000


def closest_surfaces(first, second):
    a, b, r = first
    c, d, s = second
    fit = lsq_linear(np.column_stack((b-a, c-d)), c-a, bounds=(0., 1.), tol=1e-12)
    p, q = a+fit.x[0]*(b-a), c+fit.x[1]*(d-c)
    length = np.linalg.norm(q-p)
    direction = (q-p)/length if length > 1e-12 else np.array([1., 0., 0.])
    return (p+r*direction)*1000, (q-s*direction)*1000


def render(config, q, output):
    model = config.build_kinematics()
    if any(np.any(link.opening_axis) for link in model.links):
        raise ValueError('Width-dependent geometry requires a recorded gripper width; not assumed here')
    distances = model.distances(q)
    capsules = model._capsules(model._chain(q))
    fig = plt.figure(figsize=(18, 7), facecolor='white')
    for panel, frames in enumerate((set(range(7)), {0, 2}, {4, 6}), 1):
        ax = fig.add_subplot(1, 3, panel, projection='3d')
        selected = [i for i, link in enumerate(model.links) if link.frame in frames]
        bounds = []
        for i in selected:
            a, b, r = capsules[i]
            xyz = surface(a, b, r)
            ax.plot_surface(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2],
                            color=COLORS[model.links[i].frame], alpha=.55,
                            linewidth=.15, edgecolor=(0, 0, 0, .08), shade=True)
            bounds.extend(((np.minimum(a, b)-r)*1000, (np.maximum(a, b)+r)*1000))
        if panel > 1:
            candidates = [i for i, (kind, a, b) in enumerate(model._pairs)
                          if kind == 'self' and a in selected and b in selected]
            index = min(candidates, key=lambda i: distances[i])
            _, a, b = model._pairs[index]
            p, r = closest_surfaces(capsules[a], capsules[b])
            ax.plot(*np.array([p, r]).T, color='black', linewidth=3, marker='o', markersize=4)
            title = 'Base / upper arm' if panel == 2 else 'Wrist 1 / wrist 3'
            ax.set_title(f'{title}\nMinimum capsule gap: {distances[index]*1000:.2f} mm', fontsize=12)
        else:
            ax.set_title('Recorded failing pose\nConfigured capsule geometry only', fontsize=12)
        bounds = np.array(bounds)
        center = (bounds.min(axis=0)+bounds.max(axis=0))/2
        extent = max(np.ptp(bounds, axis=0))*.58
        ax.set_xlim(center[0]-extent, center[0]+extent)
        ax.set_ylim(center[1]-extent, center[1]+extent)
        ax.set_zlim(center[2]-extent, center[2]+extent)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-55)
        ax.set_xlabel('World X (mm)', fontsize=9)
        ax.set_ylabel('World Y (mm)', fontsize=9)
        ax.set_zlabel('World Z (mm)', fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle(f'UR3e pickup clearance audit | required gap: {config.mpc["clearance"]*1000:.1f} mm', fontsize=17)
    fig.legend([Patch(color=c) for c in COLORS], LABELS, loc='lower center', ncol=7, bbox_to_anchor=(.5, .085))
    fig.text(.5, .035, 'Black segments mark nearest capsule surfaces. Padding is included; the required gap is not drawn.\n'
             'Bare arm only: gripper, table and cube are not represented. Offline visualization, not motion approval.',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.02, right=.96, top=.86, bottom=.21, wspace=.13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joints', type=float, nargs=6, default=FAILING_Q)
    parser.add_argument('--output', type=Path, default=ROOT/'simulation_plots/failing_capsule_pose.png')
    args = parser.parse_args()
    render(HardwareConfig.load(ROOT/'config'/'hardware_config.json'), args.joints, args.output)
    print(args.output)

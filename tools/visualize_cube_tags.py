"""Draw configured +Z/+Y or +Z/+X ArUco mounts, without hardware access."""
import argparse
import os
from pathlib import Path
import json

os.environ.setdefault('MPLCONFIGDIR', '/tmp/robotic-studio-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', choices=('y', 'x'), default='y')
    side = parser.parse_args().side
    marker_id = 2 if side == 'y' else 3
    config = json.loads((ROOT/'config/hardware_config.json').read_text())['perception']
    half = config['cube_size']*500
    size = config['marker_size']*1000
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    images = {i: cv2.aruco.generateImageMarker(dictionary, i, 120) for i in (0, marker_id)}
    fig = plt.figure(figsize=(13, 8), facecolor='white')
    ax = fig.add_axes([.01, .12, .61, .76], projection='3d', computed_zorder=False)
    # Cube edges, in millimeters.
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for a in (-half, half):
            for b in (-half, half):
                points = np.zeros((2, 3))
                points[:, axis] = [-half, half]
                points[:, others[0]] = a
                points[:, others[1]] = b
                ax.plot(*points.T, color='#87929a', linewidth=1.1)
    for marker in (0, marker_id):
        pose = np.array(config['cube_from_markers'][str(marker)])
        pose[:3, 3] *= 1000
        r, center = pose[:3, :3], pose[:3, 3]
        u, v = np.meshgrid(np.linspace(-size/2, size/2, 7), np.linspace(size/2, -size/2, 7))
        points = center + u[..., None]*r[:, 0] + v[..., None]*r[:, 1] + .1*r[:, 2]
        cells = images[marker][10::20, 10::20]/255.
        colors = np.repeat(cells[..., None], 4, axis=2)
        colors[..., 3] = 1
        ax.plot_surface(points[..., 0], points[..., 1], points[..., 2],
                        facecolors=colors, shade=False, antialiased=False)
        for k, (color, label) in enumerate(zip(('#c52d38', '#18864b', '#2566cc'), ('x', 'y', 'z'))):
            start = center+1*r[:, 2]
            delta = 17*r[:, k]
            ax.quiver(*start, *delta, color=color, linewidth=2, arrow_length_ratio=.2, zorder=10)
            ax.text(*(start+delta*1.25), f'{label}{marker}', color=color, fontsize=12, weight='bold', zorder=11)
    ax.set(xlim=(-45, 45), ylim=(-45, 45), zlim=(-45, 55))
    ax.set_box_aspect((1, 1, 1.1))
    ax.view_init(elev=25, azim=55 if side == 'y' else 25)
    ax.set_axis_off()
    ax.set_title('Configured mounts on the same cube', fontsize=14)
    for marker, rect, title, rotate in (
        (0, [.67, .56, .22, .25], '+Z face: ID 0', 0),
        (marker_id, [.67, .18, .22, .25], f'+{side.upper()} face: ID {marker_id}', 2 if side == 'y' else -1)):
        face = fig.add_axes(rect)
        face.imshow(np.rot90(images[marker], rotate), cmap='gray', vmin=0, vmax=255)
        face.set_xticks([])
        face.set_yticks([])
        face.set_title(title, fontsize=13, pad=24)
        face.text(.5, 1.03, 'cube +Y' if marker == 0 else 'cube +Z', transform=face.transAxes, ha='center')
        face.text(1.04, .5, 'cube +X' if marker == 0 else ('cube -X' if side == 'y' else 'cube +Y'), transform=face.transAxes, va='center')
        face.text(.5, -.16, 'Viewed from outside, looking into that face',
                  transform=face.transAxes, ha='center', fontsize=9)
    fig.suptitle(f'+{side.upper()} tag orientation relative to the +Z tag', fontsize=19, y=.97)
    fig.text(.04, .11, 'ID 0 axes:  x0 = +X    y0 = +Y    z0 = +Z\n'
             + ('ID 2 axes:  x2 = +X    y2 = -Z    z2 = +Y' if side == 'y'
                else 'ID 3 axes:  x3 = -Z    y3 = +Y    z3 = +X'), fontsize=13, linespacing=1.7)
    description = ('ID 2 is rotated -90 degrees about cube +X relative to ID 0. '
                   'The top edge of ID 2 points toward cube -Z.' if side == 'y' else
                   'ID 3 is rotated +90 degrees about cube +Y relative to ID 0. '
                   'The top edge of ID 3 points toward cube +Y.')
    fig.text(.04, .025, description + '\nThis shows the configuration, not a verification of the physical sticker mounting.', fontsize=11)
    output = ROOT/'simulation_plots'/('cube_tag_orientation.png' if side == 'y' else 'cube_tag_orientation_zx.png')
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()

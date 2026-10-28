"""Derive enclosing capsules from bundled UR3e collision STL meshes, offline."""
import argparse
import json
from pathlib import Path, PureWindowsPath
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hardware_config import HardwareConfig
from calibrate_camera import atomic_json

MODEL = ROOT.parent / 'UR3_RG2_rosbridge/models/ur3e_official'
LINKS = ['base_link_inertia', 'shoulder_link', 'upper_arm_link', 'forearm_link',
         'wrist_1_link', 'wrist_2_link', 'wrist_3_link']


def origin(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


def chain(root, q):
    poses = {'world': np.eye(4)}
    joints = root.findall('joint')
    moving = [j for j in joints if j.get('type') in ('revolute', 'continuous')]
    angles = {j.get('name'): angle for j, angle in zip(moving, q)}
    while joints:
        count = len(joints)
        for joint in joints[:]:
            parent = joint.find('parent').get('link')
            if parent not in poses:
                continue
            t = origin(joint.find('origin'))
            if joint.get('name') in angles:
                r = np.eye(4)
                r[:3, :3] = Rotation.from_rotvec(np.fromstring(joint.find('axis').get('xyz'), sep=' ') * angles[joint.get('name')]).as_matrix()
                t = t @ r
            poses[joint.find('child').get('link')] = poses[parent] @ t
            joints.remove(joint)
        if len(joints) == count:
            raise ValueError('URDF tree could not be resolved')
    base = np.linalg.inv(poses['base'])
    return {name: base @ pose for name, pose in poses.items()}


def vertices(path):
    data = path.read_bytes()
    count = int.from_bytes(data[80:84], 'little')
    if len(data) != 84 + 50 * count:
        raise ValueError(f'expected binary STL: {path}')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    return np.frombuffer(data, dtype=dtype, offset=84)['vertices'].reshape(-1, 3).astype(float)


def fit(points, margin=.003):
    center = points.mean(axis=0)
    axes = np.linalg.svd(points-center, full_matrices=False)[2]
    candidates = []
    for axis in axes:
        along = (points-center) @ axis
        low, high = along.min(), along.max()
        for trim in np.linspace(0, .5, 11):
            start = center + (low + trim*(high-low))*axis
            end = center + (high - trim*(high-low))*axis
            radius = float(segment_distances(points, start, end).max() + margin)
            candidates.append((volume(start, end, radius), start, end, radius))
    return min(candidates, key=lambda item: item[0])[1:]


def segment_distances(points, start, end):
    segment = end-start
    length_squared = segment @ segment
    fraction = (np.clip((points-start) @ segment / length_squared, 0, 1)
                if length_squared > 1e-20 else np.zeros(len(points)))
    return np.linalg.norm(points-start-fraction[:, None]*segment, axis=1)


def volume(start, end, radius):
    return np.pi * radius**2 * np.linalg.norm(end-start) + 4*np.pi*radius**3/3


def clip_polygon(polygon, axis, offset):
    """Keep the half-space dot(point, axis) >= offset, including its boundary."""
    result = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = a @ axis-offset, b @ axis-offset
        if da >= 0:
            result.append(a)
        if (da >= 0) != (db >= 0):
            result.append(a + (da/(da-db))*(b-a))
    return np.asarray(result).reshape(-1, 3)


def partition_triangles(triangles, axis, count):
    """Split triangles at slab boundaries without deleting crossing surfaces."""
    if count == 1:
        return [triangles.reshape(-1, 3)]
    projection = triangles @ axis
    boundaries = np.linspace(projection.min(), projection.max(), count+1)
    groups = [[] for _ in range(count)]
    for triangle, values in zip(triangles, projection):
        first = min(count-1, np.searchsorted(boundaries[1:-1], values.min(), side='right'))
        last = min(count-1, np.searchsorted(boundaries[1:-1], values.max(), side='right'))
        for index in range(first, last+1):
            polygon = triangle
            if index > 0:
                polygon = clip_polygon(polygon, axis, boundaries[index])
            if index < count-1 and len(polygon):
                polygon = clip_polygon(polygon, -axis, -boundaries[index+1])
            for k in range(1, len(polygon)-1):
                groups[index].extend((polygon[0], polygon[k], polygon[k+1]))
    return [np.asarray(group) for group in groups if group]


def coarse_parts(points):
    # Split triangles geometrically, not by dropping vertices outside a slab.
    # Convexity of each capsule guarantees coverage of each resulting face.
    triangles = points.reshape(-1, 3, 3)
    center = points.mean(axis=0)
    axes = np.linalg.svd(points-center, full_matrices=False)[2][:1]
    candidates = []
    lower, upper = points.min(axis=0), points.max(axis=0)
    padding = np.linalg.norm(upper-lower)
    lower, upper = lower-padding, upper+padding
    probes = np.random.default_rng(17).uniform(lower, upper, (20000, 3))
    for axis in axes:
        for count in (1, 2, 4, 8):
            along = (triangles.mean(axis=1)-center) @ axis
            bins = np.searchsorted(np.linspace(along.min(), along.max(), count+1)[1:-1], along)
            groups = [triangles[bins == index].reshape(-1, 3) for index in range(count)]
            parts = [(group, fit(group)) for group in groups if len(group)]
            # Union-volume sampling only ranks candidates; exact triangle coverage
            # below is the acceptance test. Penalize complexity on equal scores.
            inside = np.zeros(len(probes), dtype=bool)
            for _, (start, end, radius) in parts:
                inside |= segment_distances(probes, start, end) <= radius
            candidates.append(((int(inside.sum()), len(parts)), parts))
    return min(candidates, key=lambda item: item[0])[1]


def contained_fit(points, parent):
    a, b, radius = parent
    candidate = fit(points)
    candidates = [parent]
    if np.max(segment_distances(np.array(candidate[:2]), a, b)) + candidate[2] <= radius:
        candidates.append(candidate)
    segment = b-a
    if segment @ segment > 1e-20:
        fraction = np.clip((points-a) @ segment / (segment @ segment), 0, 1)
        start, end = a + fraction.min()*segment, a + fraction.max()*segment
        r = float(segment_distances(points, start, end).max() + .003)
        # Collinear centerline is a subsegment, so this proves containment.
        if r <= radius:
            candidates.append((start, end, r))
    return min(candidates, key=lambda c: volume(*c))


def fit_parts(points):
    result = []
    for original, parent in coarse_parts(points):
        triangles = original.reshape(-1, 3, 3)
        axes = np.linalg.svd(original-original.mean(axis=0), full_matrices=False)[2]
        lower = np.minimum(parent[0], parent[1])-parent[2]
        upper = np.maximum(parent[0], parent[1])+parent[2]
        probes = np.random.default_rng(23).uniform(lower, upper, (20000, 3))
        best = [(original, parent)]
        best_score = (int((segment_distances(probes, *parent[:2]) <= parent[2]).sum()), 1)
        for axis in axes:
            for count in (2, 4):
                parts = [(group, contained_fit(group, parent))
                         for group in partition_triangles(triangles, axis, count)]
                inside = np.zeros(len(probes), dtype=bool)
                for _, (a, b, r) in parts:
                    inside |= segment_distances(probes, a, b) <= r
                score = (int(inside.sum()), len(parts))
                if score < best_score:
                    best_score, best = score, parts
        result.extend(best)
    return result


def derive(config):
    root = ET.parse(MODEL / 'ur3e_official.urdf').getroot()
    model = config.build_kinematics()
    poses = chain(root, np.zeros(6))
    frames = model._chain(np.zeros(6))
    world = np.asarray(config.calibration['world_from_base'])
    offsets = [np.linalg.inv(frames[i]) @ world @ poses[name] for i, name in enumerate(LINKS)]
    # Prove each mapping stays rigid across independently varied joint angles.
    for q in np.random.default_rng(7).uniform(-2, 2, (30, 6)):
        poses_q, frames_q = chain(root, q), model._chain(q)
        for i, name in enumerate(LINKS):
            if not np.allclose(frames_q[i] @ offsets[i], world @ poses_q[name], atol=1e-7):
                raise ValueError(f'URDF/DH mapping mismatch for {name}; no config written')
        if not np.allclose(frames_q[-1], world @ poses_q['tool0'], atol=1e-7):
            raise ValueError('DH flange does not match URDF tool0')
    capsules, report = [], []
    for i, name in enumerate(LINKS):
        collision = root.find(f"link[@name='{name}']/collision")
        mesh = collision.find('geometry/mesh')
        path = MODEL / 'assets' / PureWindowsPath(mesh.get('filename')).name
        points = vertices(path) * np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
        t = offsets[i] @ origin(collision.find('origin'))
        points = points @ t[:3, :3].T + t[:3, 3]
        margins = []
        parts = fit_parts(points)
        for part, (group, (a, b, radius)) in enumerate(parts):
            distances = segment_distances(group, a, b)
            margin = float(radius-distances.max())
            if margin < .003-1e-10:
                raise ValueError(f'triangle coverage failed for {name}')
            margins.append(margin)
            capsules.append(dict(name=f'ur3e_{name}_part_{part}', frame=i,
                                 start=a.tolist(), end=b.tolist(), radius=radius))
        report.append(dict(link=name, mesh=str(path.relative_to(ROOT.parent)),
                           vertices=len(points), triangles=len(points)//3, parts=len(parts),
                           minimum_triangle_margin_m=min(margins)))
    return capsules, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-config', action='store_true')
    parser.add_argument('--replace-generated', action='store_true',
                        help='Replace only the seven previously generated arm link families')
    args = parser.parse_args()
    path = ROOT / 'config' / 'hardware_config.json'
    config = HardwareConfig.load(path)
    capsules, report = derive(config)
    atomic_json(ROOT / 'arm_capsule_report.json', dict(source=str(MODEL), margin_m=.003,
        mapping_test_poses=30, ranking='Deterministic sampled union volume; not a coverage test',
        coverage='Every STL triangle enclosed with 3 mm margin; bare UR3e only, excludes adapter, RG2, fingers, cables and environment',
        links=report, capsules=capsules))
    if args.write_config:
        original = path.read_text()
        values = json.loads(original)
        existing = values['collision']['links']
        families = {'ur3e_'+name for name in LINKS}
        def generated(name):
            return name in families or any(name.startswith(family+'_part_') for family in families)
        old_names = {c['name'] for c in existing if generated(c['name'])}
        if old_names and not args.replace_generated:
            raise ValueError('generated capsules exist; use --replace-generated to replace them')
        references = [n for pair in values['collision']['excluded_self_pairs'] for n in pair]
        references += [n for obstacle in values['collision']['obstacles'] for n in obstacle.get('exclude_links', [])]
        if old_names.intersection(references):
            raise ValueError('existing exclusions refer to replaced geometry; review them before replacement')
        archive = ROOT / 'old_version'
        archive.mkdir(parents=True, exist_ok=True)
        backup = archive / (path.name + f'.before-arm-capsules-{time.time_ns()}.bak')
        with backup.open('x') as stream:
            stream.write(original)
        values['collision']['links'] = [c for c in existing if c['name'] not in old_names] + capsules
        atomic_json(path, values)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

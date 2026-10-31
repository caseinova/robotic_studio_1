"""Compare arm capsule and mesh convex-hull clearances offline; never opens hardware."""
import argparse
import json
from pathlib import Path, PureWindowsPath
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.derive_arm_capsules import ROOT, MODEL, LINKS, chain, origin
from hardware.hardware_config import HardwareConfig


def audit(config, q):
    model = config.build_kinematics()
    distances = model.distances(q)
    threshold = config.mpc['clearance']
    capsule_pairs = [dict(pair=model.collision_pair_name(i), clearance_mm=float(d*1000))
                     for i, d in enumerate(distances) if d < threshold]
    urdf = ET.parse(MODEL / 'ur3e_official.urdf').getroot()
    poses = chain(urdf, q)
    scene = ET.Element('mujoco')
    asset = ET.SubElement(scene, 'asset')
    world = ET.SubElement(scene, 'worldbody')
    for name in LINKS:
        collision = urdf.find(f"link[@name='{name}']/collision")
        mesh = collision.find('geometry/mesh')
        path = MODEL / 'assets' / PureWindowsPath(mesh.get('filename')).name
        ET.SubElement(asset, 'mesh', name=name, file=str(path), scale=mesh.get('scale', '1 1 1'))
        pose = np.asarray(config.calibration['world_from_base']) @ poses[name] @ origin(collision.find('origin'))
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        body = ET.SubElement(world, 'body', pos=' '.join(map(str, pose[:3, 3])),
                             quat=' '.join(map(str, quat)))
        ET.SubElement(body, 'geom', name=name, type='mesh', mesh=name)
    mesh_model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))
    data = mujoco.MjData(mesh_model)
    mujoco.mj_forward(mesh_model, data)
    hull_pairs = []
    for i, a in enumerate(LINKS):
        for j in range(i+2, len(LINKS)):
            b = LINKS[j]
            gap = mujoco.mj_geomDistance(mesh_model, data, mesh_model.geom(a).id,
                                        mesh_model.geom(b).id, 2., np.zeros(6))
            hull_pairs.append(dict(pair=f'{a} / {b}', clearance_mm=float(gap*1000)))
    return dict(joint_positions_rad=list(q), required_clearance_mm=threshold*1000,
                blocked_capsule_pairs=capsule_pairs, mesh_convex_hull_pairs=hull_pairs,
                note='Nominal bare-arm collision meshes only. Convex hulls may overestimate '
                     'nonconvex geometry. Mesh clearance excludes capsule fitting padding; '
                     'not a hardware safety certification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('joints', type=float, nargs=6, help='Joint angles in radians')
    args = parser.parse_args()
    print(json.dumps(audit(HardwareConfig.load(ROOT / 'config' / 'hardware_config.json'), args.joints), indent=2))

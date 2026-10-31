"""Render the RG2 in its local frame at several openings; never opens hardware."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation.rg2_model import MAX_WIDTH, linkage_positions


def render(output):
    source = ET.parse(ROOT / 'models/ur3e/ur3e.xml').getroot()
    root = ET.Element('mujoco')
    ET.SubElement(ET.SubElement(root, 'visual'), 'global', offheight='600', offwidth='480')
    ET.SubElement(root, 'compiler', angle='radian', meshdir=str(ROOT/'models/ur3e/assets'),
                  texturedir=str(ROOT/'models/ur3e'))
    for name in ('asset', 'default', 'equality', 'contact'):
        node = source.find(name)
        if name == 'contact':
            for child in list(node):
                if child.get('body1') == 'base':
                    node.remove(child)
        root.append(node)
    world = ET.SubElement(root, 'worldbody')
    world.append(source.find('.//body[@name="gripper_base"]'))
    ET.SubElement(world, 'light', pos='.5 -.5 .5', dir='-1 1 -1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .12]
    camera.distance = .38
    camera.azimuth = 180
    camera.elevation = 0
    frames = []
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(model, height=600, width=480) as renderer:
        for width in (0, .064, MAX_WIDTH):
            data.qpos[:] = linkage_positions(width)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f'{width*1000:.1f} mm opening', (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
            frames.append(frame)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.hstack(frames))


if __name__ == '__main__':
    render(sys.argv[1] if len(sys.argv) > 1 else ROOT/'simulation_plots/rg2_model.png')

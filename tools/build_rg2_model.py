"""Install the vendored RG2 geometry into the UR3e MJCF, without hardware access."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation.rg2_model import MAX_WIDTH, linkage_positions, width_to_angle


def build():
    path = ROOT / 'models/ur3e/ur3e.xml'
    tree = ET.parse(path)
    root = tree.getroot()
    asset = root.find('asset')
    for item in list(asset):
        if item.get('name', '').startswith('rg2_'):
            asset.remove(item)
    for kind in ('visual', 'collision'):
        for part in ('base_link', 'outer_knuckle', 'inner_knuckle', 'inner_finger'):
            ET.SubElement(asset, 'mesh', name=f'rg2_{kind}_{part}',
                          file=f'rg2/{kind}/{part}.stl')
    base = root.find('.//body[@name="gripper_base"]')
    base.clear()
    base.attrib.update(name='gripper_base', childclass='gripper')

    def geom(body, part, name):
        ET.SubElement(body, 'geom', name=name+'_visual', type='mesh',
                      mesh='rg2_visual_'+part, material='black' if part == 'inner_finger' else 'linkgray',
                      contype='0', conaffinity='0', mass='0', group='2')
        ET.SubElement(body, 'geom', name=name, type='mesh', mesh='rg2_collision_'+part,
                      mass='.78' if part == 'base_link' else '.05', group='3', rgba='0 0 0 0')

    geom(base, 'base_link', 'gripper_palm')
    ET.SubElement(base, 'site', name='grasp_site', pos='0 0 0.195', size='.003', group='4')
    # Match the photographed printed pattern: +Y toward fingers, normal out of +X.
    # The local body panel is x=17.60389 mm, not the mounting-ring radius.
    tag = ET.SubElement(base, 'body', name='end_effector_tag', pos='.01765389 0 .085',
                        xyaxes='0 1 0 0 0 1')
    ET.SubElement(tag, 'geom', name='end_effector_tag_board', type='box', size='.024 .024 .00005',
                  rgba='1 1 1 1', contype='0', conaffinity='0', mass='0')
    ET.SubElement(tag, 'geom', name='end_effector_aruco', type='box', pos='0 0 .00006',
                  size='.0225 .0225 .00001', material='end_effector_aruco_material',
                  contype='0', conaffinity='0', mass='0')
    ET.SubElement(tag, 'site', name='end_effector_marker_site', pos='0 0 .00007', size='.002', group='4')
    bodies = ['gripper_base']
    for side, sign in (('left', -1), ('right', 1)):
        quat = '1 0 0 0' if sign == -1 else '0 0 0 1'
        outer = ET.SubElement(base, 'body', name=f'gripper_{side}_knuckle',
                              pos=f'0 {sign*.017178} .125797', quat=quat)
        ET.SubElement(outer, 'joint', name=f'gripper_{side}_joint', axis='1 0 0')
        geom(outer, 'outer_knuckle', f'rg2_{side}_outer_geom')
        finger = ET.SubElement(outer, 'body', name=f'gripper_{side}_finger', pos='0 -.039592 .038177')
        ET.SubElement(finger, 'joint', name=f'rg2_{side}_finger_joint', axis='1 0 0', range='-.8 .8')
        geom(finger, 'inner_finger', f'gripper_{side}_geom')
        ET.SubElement(finger, 'site', name=f'rg2_{side}_pad', pos='0 .01894995 .038', size='.002', group='4')
        inner = ET.SubElement(base, 'body', name=f'rg2_{side}_inner',
                              pos=f'0 {sign*.007678} .142297', quat=quat)
        ET.SubElement(inner, 'joint', name=f'rg2_{side}_inner_joint', axis='1 0 0')
        geom(inner, 'inner_knuckle', f'rg2_{side}_inner_geom')
        bodies.extend([outer.get('name'), finger.get('name'), inner.get('name')])
    defaults = root.find('.//default[@class="gripper"]')
    defaults.find('joint').attrib.update(type='hinge', range=f'{width_to_angle(0)} {width_to_angle(MAX_WIDTH)}',
                                         damping='.05', armature='.0001')
    defaults.find('position').attrib.update(kp='8', kv='.3',
        ctrlrange=f'{width_to_angle(0)} {width_to_angle(MAX_WIDTH)}', forcerange='-1.1 1.1')
    contact = root.find('contact')
    for item in list(contact):
        if any(item.get(key, '').startswith(('gripper_', 'rg2_')) for key in ('body1', 'body2')):
            contact.remove(item)
    # The parallelogram members intentionally overlap at their pivots.
    for i, a in enumerate(bodies):
        for b in bodies[i+1:]:
            ET.SubElement(contact, 'exclude', body1=a, body2=b)
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    for item in list(equality):
        if item.get('name', '').startswith('rg2_'):
            equality.remove(item)
    for side in ('left', 'right'):
        for part, multiplier in (('finger', -1), ('inner', 1)):
            ET.SubElement(equality, 'joint', name=f'rg2_{side}_{part}_coupling',
                          joint1=f'rg2_{side}_{part}_joint', joint2=f'gripper_{side}_joint',
                          polycoef=f'0 {multiplier} 0 0 0', solref='.002 1')
    ET.SubElement(equality, 'joint', name='rg2_jaw_coupling', joint1='gripper_right_joint',
                  joint2='gripper_left_joint', polycoef='0 1 0 0 0', solref='.002 1')
    home = root.find('.//key[@name="home"]')
    for field in ('qpos', 'ctrl'):
        values = home.get(field).split()
        values[1] = '-1.8'
        home.set(field, ' '.join(values))
    home.set('qpos', ' '.join(home.get('qpos').split()[:6]) + ' ' + ' '.join(map(str, linkage_positions(MAX_WIDTH))))
    home.set('ctrl', ' '.join(home.get('ctrl').split()[:6]) + f' {width_to_angle(MAX_WIDTH)} {width_to_angle(MAX_WIDTH)}')
    ET.indent(tree, space='  ')
    tree.write(path, encoding='unicode')
    scene_path = ROOT / 'models/ur3e/scene.xml'
    scene = ET.parse(scene_path)
    cube = scene.find('.//body[@name="cube"]')
    cube.set('pos', '-0.074 0.32 0.032')
    cube.find('geom[@name="cube_geom"]').set('size', '.032 .032 .032')
    for geom in cube.findall('geom'):
        if geom.get('name', '').startswith('cube_aruco'):
            geom.set('size', '.0225 .0225 .00005')
            coords = [float(x) for x in geom.get('pos').split()]
            geom.set('pos', ' '.join(str(.0321 if x > 0 else -.0321 if x < 0 else 0) for x in coords))
    ET.indent(scene, space='  ')
    scene.write(scene_path, encoding='unicode')


if __name__ == '__main__':
    build()

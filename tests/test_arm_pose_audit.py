import numpy as np

from hardware_config import HardwareConfig
from tools.audit_arm_pose import audit


def test_reported_pickup_pose_reproduces_capsule_and_mesh_gaps():
    q = [-1.087698761616842, -1.5815416775145472, -1.6191456317901611,
         -1.5562566083720704, 1.6523456573486328, -.3832309881793421]
    result = audit(HardwareConfig.load(), q)
    gaps = {row['pair']: row['clearance_mm'] for row in result['mesh_convex_hull_pairs']}
    assert len(gaps) == 15
    np.testing.assert_allclose(gaps['base_link_inertia / upper_arm_link'], 21.872733, atol=.01)
    np.testing.assert_allclose(gaps['wrist_1_link / wrist_3_link'], 19.418684, atol=.01)
    assert result['joint_positions_rad'] == q
    model = HardwareConfig.load().build_kinematics()
    distances = model.distances(q)
    indices = [i for i, (kind, a, b) in enumerate(model._pairs)
               if kind == 'self' and model.links[a].frame == 0 and model.links[b].frame == 2]
    assert min(distances[indices]) > .015
    assert min(distances[indices]) < .020  # Refinement does not authorize this pose.

# UR3e simulation

The arm mesh assets, joint transforms, and inertias are taken from
`UR3_RG2_rosbridge/models/ur3e_official/ur3e_official.xml` in the sibling
repository at commit `ff18f272b9988f129c23d3ffd1beb2d13762a276`.
This directory is self-contained at runtime; the source license is retained
in `SOURCE_LICENSE`.

The application scene adds the existing generic parallel gripper, ArUco
markers, external RGB-D camera, floor, and cube. The gripper is not an RG2
mechanical model. `attachment_site` follows the supplied URDF's `tool0`
rotation relative to `wrist_3_link`. The base wrapper is an identity transform.

Arm actuators retain the application's affine position/velocity switching,
with force limits from the supplied UR3e joints. The provided `intvelocity`
actuators are not used. Wrist 3 has explicit +/-2*pi simulation bounds.

The home pan angle faces the positive-Y workspace. The default cube center
is [-0.074, 0.32, 0.05] meters. The fixed camera is at
[0.5852, 0.8864, 0.85] meters, closer to the smaller workspace to retain
marker resolution under depth noise. Simulation planning clearance is 15 mm plus
the existing 3 mm planning buffer: the supplied wrist 1/wrist 3 collision
meshes have about 19 mm separation at home. Hardware configuration retains
its own explicit clearance and calibration.

This switch does not change the hardware backend's CB-series UR3 DH defaults.
Those must be configured separately for the physical installation.

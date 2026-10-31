# RG2 simulation model

Meshes and linkage origins are adapted from
https://github.com/tonydle/OnRobot_ROS2_Description
at commit `29180b3fa9cba6555f3e515e789b8ccd34252fab` (MIT, see LICENSE).
Upstream credits Osaka-University-Harada-Laboratory/onrobot.
Meshes are stored under `../ur3e/assets/rg2/`; regenerate the MJCF integration
with `python tools/build_rg2_model.py` from the application directory.

This replaces the generic sliding fingers with RG2 revolute parallelogram
linkages. MuJoCo joint equalities maintain parallel pads. Opening commands
are converted geometrically into angles, not treated as slide displacements.
The installed opening limit is 100.5 mm, matching the user's measured hardware.
The grasp origin follows the midpoint of the pads as opening changes.
Pickup predicts the pad depth at cube-width closure when setting its descent
endpoint. Simulation collision checks remain enabled for the palm/linkages.
Simulation's Python default uses 0.28 m flange approach clearance for this
longer gripper. Explicit controller configuration remains authoritative;
the legacy 0.18 m default was sized for the old short sliding fingers.

ID 1 has a 40 mm black square on the broad +X body face, with marker +Y toward
the fingers and +X across the body, matching the printed pattern in the photo.
Its nominal center is at tool coordinates
(17.724, 0, 85) mm. The 0.1 mm label backing sits against the local body panel
at X=17.604 mm, with a thin texture layer to avoid z-fighting.
This is a photo-based approximation, not a calibrated mounting
transform. Hardware calibration/configuration is not modified.

This is a community RG2 model, not a manufacturer-certified dynamics model.
Masses come from the description; MuJoCo estimates inertias from collision
meshes. Actuator gains/force caps are simulation tuning. Mesh collisions use
convex hulls; internal gripper contacts are excluded for the coupled linkage.
The source base mesh includes the mounting assembly. Validate mounting,
pad geometry, friction and payload behavior before using results quantitatively.

Validation: width conversion, independent FK, open/close dynamics, marker
mounting, rendered detection, and cube contact have offline regression tests.
A headless pick-and-place run with the current controller configuration
(including its 40 mm grasp-height offset) completes with physical cube
contacts and collision checks enabled. This validates the configured scene,
not arbitrary grasp offsets or payloads; deeper grasps can bring the inner
linkages into collision with the cube.

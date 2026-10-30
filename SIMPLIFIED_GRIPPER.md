# Simplified RG2 Collision Geometry

The five `rg2_*_simplified` capsules in `config/hardware_config.json` are an
approximation, not a fit to RG2 CAD or a validated moving-link envelope.

Measured inputs: body 135 x 60 x 38 mm, first segment 45 mm overall,
second segment 55 mm overall, finger cross sections 25 x 25 mm.
The flange and configured tool frame currently coincide.

Assumptions:
- Body extends along tool +Z from 0 to 135 mm.
- Fingers are represented as straight segments along +Z: 135..180 and
  180..235 mm. Their actual angled linkage and pivot spacing are NOT modeled.
- Opening is along tool Y, with each segment center offset by
  +/-(measured width / 2 + 12.5 mm). Confirm this direction on the installation.
- Body radius is 36 mm, rounded up from half its cross-section diagonal.
  Finger radii are 18 mm, rounded up from half their cross-section diagonal.
- Centerlines span the full stated outer lengths intentionally to enclose
  the assumed rectangular segments. Rounded caps therefore extend beyond
  those lengths: the finger envelope reaches Z=253 mm, NOT 235 or 195 mm.
  This can reject descents that are physically feasible. It does not change
  the calibrated 195 mm tool-to-grasp transform or any motion target.

The capsules enclose the assumed straight rectangular shapes, but are NOT
guaranteed to enclose the actual rotating linkages, adapter or cables.
All four finger segments use the existing intentional finger/payload contact
classification. Existing collision-pair policies and clearance limits are
unchanged. Do not interpret successful configuration validation as a physical
coverage check. Inspect against the real gripper at open/closed/intermediate
widths before hardware use. Replace this approximation with measured moving
geometry if it misses actual parts; do not compensate by reducing clearance.

The table is configured as an infinite plane at world Z=0, matching the user's
confirmation that the robot base mounting surface sits directly on the table.
World and base frames currently coincide. Only the four fixed base capsules
are excluded from table checks; moving links and the gripper remain checked.
Other environment obstacles are not represented.

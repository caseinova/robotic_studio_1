# Published simulation evidence

The current selection contains twelve timestamped runs, with original plots and
CSVs copied unchanged from `simulation_plots/`. L1-L4 use low noise, H1-H4 use
high noise, and O1-O4 are the latest four high-noise trials with both EKFs off.
L and H have both EKFs enabled. Each group uses the same four initial cube
positions and the same destination.
The manifest explicitly lists the selection. Only these twelve trial directories
are retained here, alongside the manifest, summary files and these archive notes.

`new_dest_1` and `new_dest_2` are reserved for video demonstrations and excluded
from the analysis, figures and results tables. The earlier single EKF-off run
and the four trials with mismatched depth-noise settings are superseded.
The manifest records these exclusions explicitly for provenance, but their
directories are not included in this published archive. Original experiment
data outside `docs/results/` are unaffected by this cleanup.

All twelve configuration snapshots are linked to export messages in their
source logs. Diagnostic excerpts retain
phase reports, action results and completion lines, while omitting GUI warnings
and absolute paths. All twelve logs report completed pickup and placement. No
source commit was recorded, and this selected set does not establish a general
success rate. Current defaults are not substituted for the saved run settings.

Recompute from the repository root with the simulation requirements installed:

```sh
python docs/analyze_results.py
python -m unittest discover -s docs -p 'test_analyze_results.py'
```

This regenerates `summary.json`, `summary.png` and `trajectories.png` from the published CSVs. It
does not launch the robot, camera or simulator. The JSON includes CSV SHA-256
checksums, counts, per-status metrics, covariance coverage and true endpoints.
The XY endpoint metric is the horizontal distance from the last true cube
centre to the configured destination marker. It is not full release-pose error.
In these runs the release centre is 40 mm above the marker, whereas the cube
settles to approximately 31.89 mm above the table.

Metrics are sample-weighted RMS norms, not pooled across trials or resampled.
Nonfinite, unavailable and invalid observations are excluded. A latched pickup
target is analyzed separately and excluded from live cube RMSE. Missing values
are never treated as zero. Cube sample availability is not detector success,
and recording spans are not wall-clock task durations. Marginal two-sigma
coverage is evaluated only on rows with finite pose error and covariance.

Each H/O pair has identical noise, controller, sequence and scene settings.
Estimator settings differ only in the two enable switches. Both groups use
`depth_std_per_meter_squared = 0.0015` and `depth_std = 0.0006`, resolving the
earlier mismatch. The comparison tests enabling both EKFs together, rather
than independently isolating the cube filter or tool filter.

Cube and tool position and orientation RMSEs are lower in H than O in all four
layouts, while final XY offsets remain mixed. Tool position RMSE is 27.8-53.2%
lower and orientation RMSE is 57.7-63.5% lower. Cube position RMSE is 13.8-33.1%
lower and orientation RMSE is 27.2-52.5% lower during the available initial
observation windows. These are observations from selected runs, not
an independently repeated benchmark. Closed-loop paths and observation windows
differ, and a shared seed does not guarantee the same noise at corresponding
poses. Estimation error and placement accuracy must be distinguished. Original
plots can have different axis scales, so visual smoothness alone is not a
quantitative measure.

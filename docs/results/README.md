# Published simulation evidence

The eight directories contain unchanged PNG/CSV copies of all estimation plot
sets available when this results section was assembled. `manifest.json` records
their labels and provenance. Runs 1-4 are low-noise trials with different cube
and destination positions, as reported by the experimenter. Their exact saved
configurations have not been identified. Run 5 is not assigned a noise level.

The two timestamped traces are labelled H1 and H2 on the page. Their source
console logs explicitly identify the exported plot directory, so the available
simulation configuration snapshots and task request are included. Diagnostic
excerpts retain phase reports, action results and the completion line, but omit
GUI warnings and machine-specific absolute paths. No source commit was recorded
for any run. Current repository defaults are not substituted for missing run
settings.

Recompute from the repository root with the simulation requirements installed:

```sh
python docs/analyze_results.py
python -m unittest discover -s docs -p 'test_analyze_results.py'
```

This regenerates `summary.json` and `summary.png` from the published CSVs. It
does not launch the robot, camera or simulator. The JSON includes CSV SHA-256
checksums, counts, per-status metrics, covariance coverage and true endpoints.

Metrics are sample-weighted RMS norms, not pooled across trials or resampled.
Nonfinite, unavailable and invalid observations are excluded. A latched pickup
target is analyzed separately and excluded from live cube RMSE. Missing values
are never treated as zero. Cube sample availability is not detector success,
and recording spans are not wall-clock task durations. Marginal two-sigma
coverage is evaluated only on rows with finite pose error and covariance.

These are descriptive runs, not an independently repeated, paired EKF benchmark.
The original curves can have different axis scales. Their apparent smoothness
alone does not measure filtering benefit, controller accuracy or hardware safety.

# Mathematical Audit

Scope: the project page, its linked implementation, and the published result
definitions. This is a local mathematical and implementation audit, not a
hardware safety certification or a global nonlinear stability proof.

## Verified Formulations

| Formulation | Verification and scope |
| --- | --- |
| Standard-DH transform chain | Matches `hardware/hardware_kinematics.py`. World- and tool-axis geometric Jacobians agree with central finite differences, including nonidentity base and tool offsets. |
| Joint integrator | The stated controllability/observability ranks and singular values are correct for the unconstrained six-joint integrator with nonzero time step. They do not prove Cartesian reachability. |
| Cube observability | The two-observation matrix has rank 12 at nonzero time step in the stated zero-angular-rate linearization, and rank 6 at zero time step. This is not global nonlinear observability. |
| Projection and ray-plane intersection | The formulas reconstruct synthetic oblique-plane points and agree with the implemented inverse-depth plane fit for noiseless data. |
| Frame composition | The marker-to-cube inverse and world/camera chains are consistent. A shared left transform cancels exactly before independent filtering or FK substitution. |
| Face fusion | Normalized weighted position and SVD rotation projection match the code. The weights are heuristic, not statistically optimal inverse-covariance weights. |
| Cube EKF transition | Finite differences confirm the world-axis left-rotation error transition, including `E` and `h * J_l(h * omega)`. The displayed propagation matches the implementation. |
| Tool EKF transition | Finite differences confirm right-composed FK increments and the negative skew translation/rotation cross term. Covariance propagation matches the implementation. |
| Correction and reset | The pose residual, local measurement Jacobians, gain and Joseph update match the code. Finite differences confirm the positive-sign left Jacobian for orientation reset. |
| PBVS reference | The transform-error sign, separate twist limits and damped inverse match the implementation. The corrected local contraction rates account for damping. |
| MPC objective | Matches the constant, speed-scaled PBVS reference and successive velocity differences, with the first difference relative to measured joint velocity. |
| Results and metrics | Existing tests recompute published RMSEs, sample selection and endpoint metrics from CSVs. Sampled transfer values match the archived diagnostics. These are not measured settling times or command-effort integrals. |

## Corrections and Qualifications

- The hardware grasp offset is rigid. The articulated simulation grasp offset
  also depends on finger opening width.
- The depth-inlier threshold is integer-rounded, not strictly 80% for every
  possible sample count.
- The cube angular process covariance reuses the integrated white-acceleration
  block as a small-step approximation. Numerical integration confirms that it
  differs from exact rotational discretization at nonzero angular velocity.
- Tool increment noise is a tuned diagonal random-walk model. Successive FK
  increments share encoder samples, but their temporal correlations are not
  modeled. FK initialization also uses the visual measurement variances.
- Joseph form preserves positive semidefiniteness in exact arithmetic, but
  gating and residual-dependent gain clipping do not yield an exact Gaussian
  posterior. Symmetrization alone cannot repair an indefinite matrix.
- A larger measurement variance reduces measurement influence with the same
  prior. Different cube/tool priors prevent comparing their gains using the
  measurement standard deviations alone.
- The 0.83 s time constant is the ideal undamped response. Damping introduces
  direction-dependent slower rates. Singular directions have no contraction.
- Jacobian damping depends on the numerical metre/radian scaling. The stated
  sampled scalar stability condition excludes delay and nonlinear limits.
- MPC performs sampled nonlinear clearance verification, not continuous swept-
  volume verification. Recursive feasibility and global stability are not proved.
- A Gaussian marginal two-sigma interval has 95.45% coverage, whereas a 3D
  two-sigma ellipsoid has 73.85% coverage. Neither establishes actual filter
  coverage, and viewer magnification has no probabilistic interpretation.

## Reproduce the Checks

From the repository root, with the project dependencies installed:

```bash
python -m unittest discover -s docs -p "test_*.py"
```

Audit result: **17 tests passed**, including nine mathematical checks and eight
published-result checks. The browser check rendered 132 equations without errors
and found no desktop/mobile horizontal overflow.

A separate EGL-backed regression run of `test_estimator.py`,
`test_tool_increment_ekf.py`, `test_controller.py`, `test_hardware_kinematics.py`
and `test_camera_processing.py` produced **46 passes and 11 failures**. Failures
include an estimator reference mismatch, old target-height expectations,
clearance/gradient expectations, and rendered-marker detection tests. Several
fixtures still use older cube/tag dimensions. These failures remain unresolved;
the mathematical audit must not be presented as a clean integration-test result.
No runtime control or estimator code was changed during this audit.

## Public Theory References

- [Sola, Deray and Atchuthan, A micro Lie theory for state estimation in robotics](https://arxiv.org/pdf/1812.01537): left/right perturbation conventions and Jacobians.
- [Chaumette and Hutchinson, Visual Servo Control, Part I: Basic Approaches](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf): visual-servo interaction models and stability assumptions.

The page's implementation-specific Jacobians and damped local response were
also checked independently against code and numerical derivatives. The papers
do not establish correctness or stability of this particular application.

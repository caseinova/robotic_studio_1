import numpy as np


def damped_least_squares(
    J,
    xdot,
    damping=0.05,
):
    """
    Compute joint velocity from Cartesian velocity using
    Damped Least Squares (DLS).

    Parameters
    ----------
    J : array-like, shape (6, 6)
        Geometric Jacobian.

    xdot : array-like, shape (6,)
        Desired Cartesian twist:

        [
            vx, vy, vz,
            wx, wy, wz
        ]

        Linear velocity:  m/s
        Angular velocity: rad/s

    damping : float
        DLS damping factor.

    Returns
    -------
    qdot : ndarray, shape (6,)
        Joint velocity command [rad/s].
    """

    J = np.asarray(
        J,
        dtype=float,
    )

    xdot = np.asarray(
        xdot,
        dtype=float,
    )


    if J.shape != (6, 6):
        raise ValueError(
            f"Expected Jacobian shape (6, 6), "
            f"got {J.shape}"
        )


    if xdot.shape != (6,):
        raise ValueError(
            "xdot must contain exactly 6 values."
        )


    if not np.all(
        np.isfinite(J)
    ):
        raise ValueError(
            "Jacobian contains NaN or Inf."
        )


    if not np.all(
        np.isfinite(xdot)
    ):
        raise ValueError(
            "xdot contains NaN or Inf."
        )


    if damping < 0:
        raise ValueError(
            "damping must be >= 0."
        )


    # ========================================================
    # Damped Least Squares
    #
    # qdot =
    #
    # J^T (J J^T + lambda^2 I)^-1 xdot
    #
    # solve() is used instead of explicitly computing inverse.
    # ========================================================

    A = (
        J @ J.T
        + (damping ** 2)
        * np.eye(6)
    )


    qdot = (
        J.T
        @ np.linalg.solve(
            A,
            xdot,
        )
    )


    return qdot


def limit_joint_velocity(
    qdot,
    max_speed,
):
    """
    Uniformly scale qdot so no joint exceeds max_speed.

    Direction of the joint-velocity vector is preserved.
    """

    qdot = np.asarray(
        qdot,
        dtype=float,
    )


    if qdot.shape != (6,):
        raise ValueError(
            "qdot must contain exactly 6 values."
        )


    if max_speed <= 0:
        raise ValueError(
            "max_speed must be > 0."
        )


    peak = np.max(
        np.abs(qdot)
    )


    if peak > max_speed:

        scale = (
            max_speed
            / peak
        )

        qdot = (
            qdot
            * scale
        )


    return qdot
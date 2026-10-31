import numpy as np


def limit_cartesian_velocity(
    xdot,
    max_linear_speed=0.02,
    max_angular_speed=0.20,
):
    """
    Limit Cartesian twist while preserving its overall direction.

    xdot:
        [vx, vy, vz, wx, wy, wz]

    Linear units:
        m/s

    Angular units:
        rad/s
    """

    xdot = np.asarray(
        xdot,
        dtype=float,
    )

    if xdot.shape != (6,):
        raise ValueError(
            "xdot must contain exactly 6 values."
        )

    if not np.all(np.isfinite(xdot)):
        raise ValueError(
            "xdot contains NaN or Inf."
        )

    if max_linear_speed <= 0:
        raise ValueError(
            "max_linear_speed must be > 0."
        )

    if max_angular_speed <= 0:
        raise ValueError(
            "max_angular_speed must be > 0."
        )


    linear = xdot[:3]
    angular = xdot[3:]


    linear_norm = np.linalg.norm(
        linear
    )

    angular_norm = np.linalg.norm(
        angular
    )


    scale = 1.0


    if linear_norm > max_linear_speed:

        scale = min(
            scale,
            max_linear_speed
            / linear_norm,
        )


    if angular_norm > max_angular_speed:

        scale = min(
            scale,
            max_angular_speed
            / angular_norm,
        )


    return xdot * scale


def adaptive_damping(
    J,
    minimum_damping=0.05,
    maximum_damping=0.20,
    singularity_threshold=0.08,
):
    """
    Increase DLS damping when the Jacobian approaches
    a singular configuration.

    Returns
    -------
    damping : float
    sigma_min : float
    """

    J = np.asarray(
        J,
        dtype=float,
    )


    singular_values = np.linalg.svd(
        J,
        compute_uv=False,
    )


    sigma_min = float(
        np.min(
            singular_values
        )
    )


    # Far enough from singularity
    if sigma_min >= singularity_threshold:

        return (
            minimum_damping,
            sigma_min,
        )


    # Closer to singularity:
    # smoothly increase damping

    ratio = (
        singularity_threshold
        - sigma_min
    ) / singularity_threshold


    ratio = np.clip(
        ratio,
        0.0,
        1.0,
    )


    damping = (
        minimum_damping
        + ratio
        * (
            maximum_damping
            - minimum_damping
        )
    )


    return (
        float(damping),
        sigma_min,
    )


def apply_joint_limit_protection(
    q,
    qdot,
    qlim,
    margin=np.deg2rad(10.0),
):
    """
    Gradually reduce velocity when moving toward a joint limit.

    qlim shape:
        (2, 6)

        qlim[0, :] -> lower limits
        qlim[1, :] -> upper limits

    Infinite limits are ignored.
    """

    q = np.asarray(
        q,
        dtype=float,
    )

    qdot = np.asarray(
        qdot,
        dtype=float,
    ).copy()

    qlim = np.asarray(
        qlim,
        dtype=float,
    )


    if q.shape != (6,):
        raise ValueError(
            "q must contain exactly 6 values."
        )

    if qdot.shape != (6,):
        raise ValueError(
            "qdot must contain exactly 6 values."
        )

    if qlim.shape != (2, 6):
        raise ValueError(
            f"Expected qlim shape (2, 6), "
            f"got {qlim.shape}"
        )


    for i in range(6):

        lower = qlim[0, i]
        upper = qlim[1, i]


        # ----------------------------------------------------
        # Moving toward lower limit
        # ----------------------------------------------------

        if (
            np.isfinite(lower)
            and qdot[i] < 0.0
        ):

            distance = (
                q[i]
                - lower
            )


            if distance <= 0.0:

                qdot[i] = 0.0


            elif distance < margin:

                qdot[i] *= (
                    distance
                    / margin
                )


        # ----------------------------------------------------
        # Moving toward upper limit
        # ----------------------------------------------------

        if (
            np.isfinite(upper)
            and qdot[i] > 0.0
        ):

            distance = (
                upper
                - q[i]
            )


            if distance <= 0.0:

                qdot[i] = 0.0


            elif distance < margin:

                qdot[i] *= (
                    distance
                    / margin
                )


    return qdot


def limit_joint_velocity(
    qdot,
    max_speed=0.10,
):
    """
    Uniformly scale qdot so that no joint exceeds max_speed.

    Uniform scaling preserves the qdot direction.
    """

    qdot = np.asarray(
        qdot,
        dtype=float,
    )


    if qdot.shape != (6,):
        raise ValueError(
            "qdot must contain exactly 6 values."
        )


    if not np.all(np.isfinite(qdot)):
        raise ValueError(
            "qdot contains NaN or Inf."
        )


    if max_speed <= 0:
        raise ValueError(
            "max_speed must be > 0."
        )


    peak = float(
        np.max(
            np.abs(qdot)
        )
    )


    if peak <= max_speed:

        return qdot


    scale = (
        max_speed
        / peak
    )


    return (
        qdot
        * scale
    )
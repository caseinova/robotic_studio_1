import numpy as np

from rtde_receive import (
    RTDEReceiveInterface as RTDEReceive
)

from rtde_control import (
    RTDEControlInterface as RTDEControl
)

from .base_backend import RobotBackend


class RealUR3e(RobotBackend):
    """
    Real UR3e backend using ur_rtde.

    Default:
        READ ONLY

    Motion is enabled only when:
        enable_control=True

    Joint order:
        0 shoulder_pan
        1 shoulder_lift
        2 elbow
        3 wrist_1
        4 wrist_2
        5 wrist_3
    """

    def __init__(
        self,
        robot_ip,
        control_frequency=100.0,
        enable_control=False,
        max_joint_speed=0.05,
        acceleration=0.10,
        stop_acceleration=0.50,
    ):

        self.robot_ip = str(
            robot_ip
        )

        self.control_frequency = float(
            control_frequency
        )

        if self.control_frequency <= 0:

            raise ValueError(
                "control_frequency must be > 0."
            )


        self.control_dt = (
            1.0
            / self.control_frequency
        )


        self.enable_control = bool(
            enable_control
        )


        self.max_joint_speed = float(
            max_joint_speed
        )

        if self.max_joint_speed <= 0:

            raise ValueError(
                "max_joint_speed must be > 0."
            )


        self.acceleration = float(
            acceleration
        )

        self.stop_acceleration = float(
            stop_acceleration
        )


        self._closed = False

        self.rtde_c = None


        # ====================================================
        # Receive connection
        # ====================================================

        print()
        print(
            f"Connecting RTDE receive to "
            f"{self.robot_ip}..."
        )


        self.rtde_r = RTDEReceive(
            self.robot_ip
        )


        print(
            "RTDE receive connected."
        )


        # ====================================================
        # Optional control connection
        # ====================================================

        if self.enable_control:

            print()
            print(
                "Connecting RTDE control..."
            )


            # Second constructor argument is control frequency.

            try:
                self.rtde_c = RTDEControl(
                    self.robot_ip,
                    self.control_frequency,
                )
            except Exception:
                self.rtde_r.disconnect()
                raise


            print(
                "RTDE control connected."
            )


        else:

            print()
            print(
                "Backend running in READ-ONLY mode."
            )


    # ========================================================
    # Internal checks
    # ========================================================

    def _check_open(
        self,
    ):

        if self._closed:

            raise RuntimeError(
                "UR3e backend is already closed."
            )


    def _check_control_enabled(
        self,
    ):

        if not self.enable_control:

            raise RuntimeError(
                "Real UR3e motion is disabled. "
                "Create the backend with "
                "enable_control=True."
            )


        if self.rtde_c is None:

            raise RuntimeError(
                "RTDE control interface is unavailable."
            )


    # ========================================================
    # Joint positions
    # ========================================================

    def get_joint_positions(
        self,
    ):

        self._check_open()


        q = np.asarray(
            self.rtde_r.getActualQ(),
            dtype=float,
        )


        if q.shape != (6,):

            raise RuntimeError(
                f"Unexpected joint position "
                f"shape: {q.shape}"
            )


        return q


    # ========================================================
    # Joint velocities
    # ========================================================

    def get_joint_velocities(
        self,
    ):

        self._check_open()


        qd = np.asarray(
            self.rtde_r.getActualQd(),
            dtype=float,
        )


        if qd.shape != (6,):

            raise RuntimeError(
                f"Unexpected joint velocity "
                f"shape: {qd.shape}"
            )


        return qd


    # ========================================================
    # Actual active TCP
    # ========================================================

    def get_tcp_pose(
        self,
    ):

        self._check_open()


        tcp = np.asarray(
            self.rtde_r.getActualTCPPose(),
            dtype=float,
        )


        if tcp.shape != (6,):

            raise RuntimeError(
                f"Unexpected TCP pose "
                f"shape: {tcp.shape}"
            )


        return tcp


    # ========================================================
    # Backend-level velocity limiter
    #
    # This is intentionally duplicated even though the
    # controller safety layer also limits qdot.
    #
    # Defence in depth:
    # bad student/controller code still cannot exceed this.
    # ========================================================

    def _limit_joint_velocity(
        self,
        qd,
    ):

        peak = float(
            np.max(
                np.abs(qd)
            )
        )


        if peak <= self.max_joint_speed:

            return qd


        scale = (
            self.max_joint_speed
            / peak
        )


        return (
            qd
            * scale
        )


    # ========================================================
    # Main motion command
    #
    # Same public API as MuJoCoUR3e:
    #
    #     robot.set_joint_velocity(qd)
    #
    # ========================================================

    def set_joint_velocity(
        self,
        qd,
    ):

        self._check_open()
        self._check_control_enabled()


        qd = np.asarray(
            qd,
            dtype=float,
        )


        if qd.shape != (6,):

            raise ValueError(
                "qd must contain exactly "
                "6 joint velocities."
            )


        if not np.all(
            np.isfinite(qd)
        ):

            self.stop()

            raise ValueError(
                "qd contains NaN or Inf."
            )


        # ----------------------------------------------------
        # Backend hard velocity limit
        # ----------------------------------------------------

        qd_safe = (
            self._limit_joint_velocity(
                qd
            )
        )


        # ----------------------------------------------------
        # Begin one RTDE control period
        # ----------------------------------------------------

        t_start = (
            self.rtde_c.initPeriod()
        )


        try:

            success = (
                self.rtde_c.speedJ(
                    qd_safe.tolist(),
                    self.acceleration,
                    self.control_dt,
                )
            )


            if not success:

                self.stop()

                raise RuntimeError(
                    "UR speedJ command failed."
                )


        except Exception:

            try:

                self.rtde_c.speedStop(
                    self.stop_acceleration
                )

            except Exception:

                pass

            raise


        # ----------------------------------------------------
        # Complete timing period
        # ----------------------------------------------------

        self.rtde_c.waitPeriod(
            t_start
        )


    # ========================================================
    # Stop joint-speed mode
    # ========================================================

    def stop(
        self,
    ):

        if self._closed:
            return


        if (
            self.enable_control
            and self.rtde_c is not None
        ):

            try:

                self.rtde_c.speedStop(
                    self.stop_acceleration
                )

            except Exception:

                pass


    # ========================================================
    # Close backend
    # ========================================================

    def close(
        self,
    ):

        if self._closed:
            return


        # ----------------------------------------------------
        # Stop robot motion first
        # ----------------------------------------------------

        self.stop()


        # ----------------------------------------------------
        # Stop RTDE control script
        # ----------------------------------------------------

        if self.rtde_c is not None:

            try:

                self.rtde_c.stopScript()

            except Exception:

                pass


            try:

                self.rtde_c.disconnect()

            except Exception:

                pass


            self.rtde_c = None


        # ----------------------------------------------------
        # Disconnect receiver
        # ----------------------------------------------------

        try:

            self.rtde_r.disconnect()

        except Exception:

            pass


        self._closed = True


        print(
            "UR3e RTDE connections closed."
        )

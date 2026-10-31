from pathlib import Path
import time

import numpy as np

import mujoco
import mujoco.viewer

from robot.base_backend import RobotBackend


# ============================================================
# UR3e official joint ordering
#
# IMPORTANT:
# Keep this same ordering throughout the entire project.
#
# Later:
#   RTDE getActualQ()
#   RTDE speedJ()
#
# will use the same conceptual order.
# ============================================================

UR3E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


UR3E_ACTUATOR_NAMES = [
    "ur3e_qd_shoulder_pan_joint",
    "ur3e_qd_shoulder_lift_joint",
    "ur3e_qd_elbow_joint",
    "ur3e_qd_wrist_1_joint",
    "ur3e_qd_wrist_2_joint",
    "ur3e_qd_wrist_3_joint",
]


class MuJoCoUR3e(RobotBackend):

    def __init__(
        self,
        model_path,
        control_frequency=100.0,
        show_viewer=True,
        realtime=True,
    ):

        # ====================================================
        # General settings
        # ====================================================

        self.model_path = Path(
            model_path
        )

        self.control_frequency = float(
            control_frequency
        )

        if self.control_frequency <= 0:

            raise ValueError(
                "control_frequency must be > 0"
            )

        self.control_dt = (
            1.0
            / self.control_frequency
        )

        self.realtime = bool(
            realtime
        )


        # ====================================================
        # Check model
        # ====================================================

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"MuJoCo model not found:\n"
                f"{self.model_path}"
            )


        # ====================================================
        # Load MuJoCo model
        # ====================================================

        self.model = (
            mujoco.MjModel.from_xml_path(
                str(self.model_path)
            )
        )

        self.data = mujoco.MjData(
            self.model
        )


        # ====================================================
        # Find six UR3e joints
        # ====================================================

        self.joint_ids = []
        self.qpos_addresses = []
        self.dof_addresses = []


        for joint_name in UR3E_JOINT_NAMES:

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if joint_id < 0:

                raise RuntimeError(
                    f"UR3e joint not found: "
                    f"{joint_name}"
                )


            qpos_address = int(
                self.model.jnt_qposadr[
                    joint_id
                ]
            )

            dof_address = int(
                self.model.jnt_dofadr[
                    joint_id
                ]
            )


            self.joint_ids.append(
                joint_id
            )

            self.qpos_addresses.append(
                qpos_address
            )

            self.dof_addresses.append(
                dof_address
            )


        # ====================================================
        # Find six velocity actuators
        # ====================================================

        self.actuator_ids = []


        for actuator_name in UR3E_ACTUATOR_NAMES:

            actuator_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )

            if actuator_id < 0:

                raise RuntimeError(
                    f"UR3e actuator not found: "
                    f"{actuator_name}"
                )

            self.actuator_ids.append(
                actuator_id
            )


        # ====================================================
        # Reset simulation
        # ====================================================

        mujoco.mj_resetData(
            self.model,
            self.data,
        )


        # ====================================================
        # Initialise integrated-velocity actuator targets
        #
        # intvelocity stores an internal position target.
        # Initial target should equal the current joint angle.
        # ====================================================

        self._initialise_actuator_targets()


        # ====================================================
        # Initial velocity command = zero
        # ====================================================

        self._write_velocity_command(
            np.zeros(6)
        )


        # ====================================================
        # Compute initial forward state
        # ====================================================

        mujoco.mj_forward(
            self.model,
            self.data,
        )


        # ====================================================
        # Viewer
        # ====================================================

        self.viewer = None

        if show_viewer:

            self.viewer = (
                mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                )
            )

            self.viewer.sync()


        # ====================================================
        # Closed state
        # ====================================================

        self._closed = False


    # ========================================================
    # Internal:
    # initialise intvelocity actuator position targets
    # ========================================================

    def _initialise_actuator_targets(
        self,
    ):

        for (
            joint_id,
            actuator_id,
        ) in zip(
            self.joint_ids,
            self.actuator_ids,
        ):

            qpos_address = int(
                self.model.jnt_qposadr[
                    joint_id
                ]
            )

            act_address = int(
                self.model.actuator_actadr[
                    actuator_id
                ]
            )


            # intvelocity has activation state.
            #
            # Some actuator types do not.
            # Therefore check >= 0.

            if act_address >= 0:

                self.data.act[
                    act_address
                ] = self.data.qpos[
                    qpos_address
                ]


    # ========================================================
    # Internal:
    # write six qdot commands
    # ========================================================

    def _write_velocity_command(
        self,
        qd,
    ):

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

            raise ValueError(
                "qd contains NaN or Inf."
            )


        # IMPORTANT:
        #
        # In MuJoCo, data.ctrl is indexed directly
        # by actuator ID.
        #
        # There is no separate ctrl address needed here.

        for (
            actuator_id,
            command,
        ) in zip(
            self.actuator_ids,
            qd,
        ):

            self.data.ctrl[
                actuator_id
            ] = command


    # ========================================================
    # Feedback:
    # q [rad]
    # ========================================================

    def get_joint_positions(
        self,
    ):

        return np.array(
            [
                self.data.qpos[
                    address
                ]
                for address
                in self.qpos_addresses
            ],
            dtype=float,
        )


    # ========================================================
    # Feedback:
    # qdot [rad/s]
    # ========================================================

    def get_joint_velocities(
        self,
    ):

        return np.array(
            [
                self.data.qvel[
                    address
                ]
                for address
                in self.dof_addresses
            ],
            dtype=float,
        )


    # ========================================================
    # Main command:
    # qdot [rad/s]
    # ========================================================

    def set_joint_velocity(
        self,
        qd,
    ):

        if self._closed:

            raise RuntimeError(
                "MuJoCo backend is already closed."
            )


        cycle_start = (
            time.perf_counter()
        )


        # ----------------------------------------------------
        # Send velocity command
        # ----------------------------------------------------

        self._write_velocity_command(
            qd
        )


        # ----------------------------------------------------
        # Advance physics by one control period
        #
        # Using simulation time instead of assuming that
        # control_dt is exactly divisible by physics timestep.
        # ----------------------------------------------------

        target_sim_time = (
            self.data.time
            + self.control_dt
        )


        while (
            self.data.time
            < target_sim_time
        ):

            mujoco.mj_step(
                self.model,
                self.data,
            )


        # ----------------------------------------------------
        # Update viewer
        # ----------------------------------------------------

        if self.viewer is not None:

            if self.viewer.is_running():

                self.viewer.sync()


        # ----------------------------------------------------
        # Optional real-time pacing
        # ----------------------------------------------------

        if self.realtime:

            elapsed = (
                time.perf_counter()
                - cycle_start
            )

            remaining = (
                self.control_dt
                - elapsed
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )


    # ========================================================
    # Stop motion
    # ========================================================

    def stop(
        self,
    ):

        if self._closed:
            return

        self._write_velocity_command(
            np.zeros(6)
        )


    # ========================================================
    # Reset simulation
    # ========================================================

    def reset(
        self,
        q=None,
    ):

        if self._closed:

            raise RuntimeError(
                "MuJoCo backend is already closed."
            )


        # Reset entire MuJoCo state
        mujoco.mj_resetData(
            self.model,
            self.data,
        )


        # --------------------------------------------------------
        # Optional initial UR3e configuration
        # --------------------------------------------------------

        if q is not None:

            q = np.asarray(
                q,
                dtype=float,
            )

            if q.shape != (6,):

                raise ValueError(
                    "q must contain exactly "
                    "6 joint positions."
                )


            if not np.all(
                np.isfinite(q)
            ):

                raise ValueError(
                    "q contains NaN or Inf."
                )


            for (
                address,
                value,
            ) in zip(
                self.qpos_addresses,
                q,
            ):

                self.data.qpos[
                    address
                ] = value


        # --------------------------------------------------------
        # Initialise intvelocity targets from current q
        # --------------------------------------------------------

        self._initialise_actuator_targets()


        # Zero velocity command
        self._write_velocity_command(
            np.zeros(6)
        )


        mujoco.mj_forward(
            self.model,
            self.data,
        )


        if self.viewer is not None:

            if self.viewer.is_running():

                self.viewer.sync()


    # ========================================================
    # Close simulator
    # ========================================================

    def close(
        self,
    ):

        if self._closed:
            return


        # Stop command first
        self.stop()


        if self.viewer is not None:

            try:

                self.viewer.close()

            except Exception:

                pass


            self.viewer = None


        self._closed = True
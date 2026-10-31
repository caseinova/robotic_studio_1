"""Verify the controller boundary using an interface with no live simulator."""

from dataclasses import replace
import unittest

import numpy as np

from common.controller import ApproachController
from common.controller_config import ControllerConfig
from common.interface import RobotInterface
from simulation.kinematics import MujocoKinematics
from simulation.robot_model import RobotModel
from common.robot_state import RobotState
from simulation.simulate import Simulation


class RecordedRobotInterface(RobotInterface):
    """A hardware-shaped adapter replaying one measurement and camera frame."""

    def __init__(self, state, frame):
        self.state = state
        self.frame = frame
        self.now = state.timestamp
        self.state_reads = 0
        self.camera_reads = 0
        self.commands = []
        self.width_commands = []

    @property
    def timestamp(self):
        return self.now

    @property
    def gripper_max_width(self):
        return 0.085

    def read_state(self):
        self.state_reads += 1
        return self.state

    def capture_rgbd(self):
        self.camera_reads += 1
        return self.frame

    def set_joint_positions(self, positions):
        self.commands.append(np.array(positions))

    def set_joint_velocities(self, velocities):
        self.commands.append(np.array(velocities))

    def set_gripper_width(self, width):
        self.width_commands.append(width)

    def close(self):
        pass


class InterfaceTests(unittest.TestCase):
    def test_controller_uses_interface_measurements_with_no_live_robot_data(self):
        robot = RobotModel()
        frame = replace(robot.capture_rgbd(), timestamp=1234.0)
        positions = robot.joint_positions
        positions[0] += 0.08
        state = RobotState(1234.0, positions, np.zeros(6), 0.04)
        # The model definition remains usable, but live simulator data is gone.
        robot.data = None
        for mode in ("pure_pbvs", "pbvs_mpc"):
            with self.subTest(mode=mode):
                interface = RecordedRobotInterface(state, frame)
                kinematics = MujocoKinematics(robot)
                controller = ApproachController(
                    interface, kinematics, controller_config=ControllerConfig(mode=mode)
                )
                observation = controller.processor.process(frame)
                kinematics.synchronize(state)
                tool_observation = controller.tool_processor.process_marker(frame)
                camera_from_tool = (tool_observation.camera_from_marker
                                    @ np.linalg.inv(kinematics.tool_from_end_effector_marker))
                expected_reference, _ = controller.pbvs.reference(
                    np.linalg.inv(camera_from_tool) @ observation.camera_from_marker,
                    kinematics.tool_jacobian(positions, local=True),
                )
                result = controller.update()
                self.assertEqual(result.status, "tracking")
                self.assertEqual(interface.state_reads, 1)
                self.assertEqual(interface.camera_reads, 1)
                np.testing.assert_allclose(result.reference_velocity, expected_reference)
                np.testing.assert_array_equal(interface.commands[-1], result.command_velocity)
                np.testing.assert_allclose(kinematics.data.qpos[kinematics.gripper_indices], 0.02)

    def test_stale_camera_or_state_stops_using_interface_clock(self):
        robot = RobotModel()
        frame = replace(robot.capture_rgbd(), timestamp=100.0)
        state = RobotState(100.0, robot.joint_positions, np.zeros(6), 0.12)
        for stale_field, diagnostic in (
            ("frame", "stale_rgbd_frame"), ("state", "stale_robot_state")
        ):
            interface = RecordedRobotInterface(state, frame)
            if stale_field == "frame":
                interface.frame = replace(frame, timestamp=99.0)
            else:
                interface.state = replace(state, timestamp=99.0)
            controller = ApproachController(interface, MujocoKinematics(robot))
            result = controller.update()
            self.assertEqual(result.status, "target_lost_or_stale")
            self.assertEqual(result.diagnostic, diagnostic)
            np.testing.assert_array_equal(interface.commands[-1], np.zeros(6))

    def test_simulation_snapshot_camera_and_resource_lifetime(self):
        simulation = Simulation()
        with simulation.interface as interface:
            state = interface.read_state()
            frame = interface.capture_rgbd()
            self.assertEqual(frame.timestamp, state.timestamp)
            saved_rgb = frame.rgb.copy()
            state.joint_positions[:] = 0
            state.joint_velocities[:] = 100
            self.assertFalse(np.all(interface.joint_positions == 0))
            np.testing.assert_array_equal(interface.joint_velocities, np.zeros(6))
            simulation.step(25)
            second = interface.capture_rgbd()
            self.assertGreater(second.timestamp, frame.timestamp)
            self.assertEqual(second.timestamp, interface.timestamp)
            np.testing.assert_array_equal(frame.rgb, saved_rgb)
        self.assertIsNone(interface._renderer)
        interface.close()  # Repeated cleanup is harmless.

    def test_open_gripper_uses_backend_width(self):
        state = RobotState(0.0, np.zeros(6), np.zeros(6), 0.0)
        interface = RecordedRobotInterface(state, None)
        interface.open_gripper()
        interface.close_gripper()
        self.assertEqual(interface.width_commands, [0.085, 0.0])


if __name__ == "__main__":
    unittest.main()

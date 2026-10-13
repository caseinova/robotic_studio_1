"""Telemetry frame/velocity checks and real simulation PNG/CSV export."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from controller import ControlResult
from controller_config import ControllerConfig
from kinematics import MujocoKinematics
from plots import SimulationPlots
from robot_model import RobotModel
from robot_state import RobotState
from simulate import Simulation


class PlotTests(unittest.TestCase):
    def test_measured_camera_twist_matches_pose_derivative_and_preserves_sources(self):
        robot = RobotModel()
        kinematics = MujocoKinematics(robot)
        q = robot.joint_positions
        dq = np.array([0.02, -0.01, 0.03, 0.01, -0.02, 0.01])
        state = RobotState(4.0, q.copy(), dq.copy(), 0.12)
        result = ControlResult(None, 2 * dq, 0.5 * dq, "tracking", 0.1, 0.1, np.nan)
        # FK/Jacobians must use the measurement, not simulator state.
        robot.data = None
        recorder = SimulationPlots()
        recorder.record(state, kinematics, result)
        data = recorder.data
        epsilon = 1e-5
        kinematics.set_positions(q + epsilon * dq)
        plus = kinematics.camera_pose()
        kinematics.set_positions(q - epsilon * dq)
        minus = kinematics.camera_pose()
        camera_velocity = (plus[:3, 3] - minus[:3, 3]) / (2 * epsilon)
        rotation_rate = (plus[:3, :3] - minus[:3, :3]) / (2 * epsilon)
        kinematics.set_positions(q)
        omega_cross = rotation_rate @ kinematics.camera_pose()[:3, :3].T
        camera_omega = np.array([omega_cross[2, 1], omega_cross[0, 2], omega_cross[1, 0]])
        np.testing.assert_allclose(data["linear_velocity"][0], camera_velocity, atol=1e-8)
        np.testing.assert_allclose(data["angular_velocity"][0], camera_omega, atol=1e-8)
        np.testing.assert_allclose(data["reference_linear_velocity"][0], 2 * camera_velocity, atol=1e-8)
        np.testing.assert_allclose(data["command_angular_velocity"][0], 0.5 * camera_omega, atol=1e-8)
        np.testing.assert_array_equal(state.joint_positions, q)
        data["camera_position"][:] = 0
        self.assertFalse(np.all(recorder.data["camera_position"] == 0))

    def test_pickup_reference_time_reset_and_duplicate_samples(self):
        simulation = Simulation()
        kinematics = MujocoKinematics(simulation.robot)
        state = simulation.interface.read_state()
        result = ControlResult(None, np.ones(6) * 0.01, np.zeros(6), "pickup_lifting", 0, 0, np.nan)
        recorder = SimulationPlots()
        recorder.record(state, kinematics, result)
        recorder.record(state, kinematics, result)
        self.assertEqual(len(recorder.data["time"]), 1)
        self.assertTrue(np.isnan(recorder.data["reference_linear_velocity"]).all())
        simulation.step(25)
        recorder.record(simulation.interface.read_state(), kinematics, result)
        np.testing.assert_allclose(recorder.data["time"], [0, 0.05])
        with self.assertRaises(ValueError):
            recorder.record(state, kinematics, result)
        recorder.reset()
        self.assertEqual(len(recorder.data["time"]), 0)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(recorder.save(directory), [])

    def test_headless_simulation_records_and_exports_without_pyplot(self):
        simulation = Simulation(controller_config=ControllerConfig(mode="pure_pbvs"))
        result = simulation.run_controlled(100)
        self.assertEqual(result.status, "tracking")
        data = simulation.plots.data
        self.assertEqual(len(data["time"]), 4)
        self.assertGreater(np.linalg.norm(data["camera_position"][-1] - data["camera_position"][0]), 0.001)
        with tempfile.TemporaryDirectory() as directory:
            paths = simulation.plots.save(directory)
            self.assertEqual({path.name for path in paths}, {
                "trajectory.png", "linear_velocity.png", "angular_velocity.png", "trajectory.csv",
            })
            for path in paths:
                self.assertGreater(path.stat().st_size, 100)
            with (Path(directory) / "trajectory.png").open("rb") as source:
                self.assertEqual(source.read(8), b"\x89PNG\r\n\x1a\n")
            with (Path(directory) / "trajectory.csv").open(newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 4)
            np.testing.assert_allclose([float(row["time_s"]) for row in rows], data["time"])
        simulation.reset()
        self.assertEqual(len(simulation.plots.data["time"]), 0)
        simulation.plots.close()

    def test_viewer_records_and_refreshes_when_plots_enabled(self):
        simulation = Simulation(controller_config=ControllerConfig(mode="pure_pbvs"))
        viewer = MagicMock()
        viewer.is_running.side_effect = [True, True, False]
        context = MagicMock()
        context.__enter__.return_value = viewer
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch("mujoco.viewer.launch_passive", return_value=context))
            for name in ("namedWindow", "imshow", "destroyWindow"):
                stack.enter_context(patch(f"cv2.{name}"))
            stack.enter_context(patch("cv2.waitKey", return_value=-1))
            stack.enter_context(patch("cv2.getWindowProperty", return_value=1))
            stack.enter_context(patch("simulate.time.sleep"))
            live = stack.enter_context(patch.object(simulation.plots, "start_live"))
            refresh = stack.enter_context(patch.object(simulation.plots, "refresh"))
            simulation.launch_viewer(servo=True, plots=True)
        live.assert_called_once()
        self.assertGreaterEqual(refresh.call_count, 1)
        self.assertEqual(len(simulation.plots.data["time"]), 2)

    def test_live_figures_can_be_closed_independently_and_exported(self):
        import matplotlib
        matplotlib.use("Agg")  # Exercise live lifecycle without opening GUI windows.
        from matplotlib import pyplot as plt

        simulation = Simulation(controller_config=ControllerConfig(mode="pure_pbvs"))
        simulation.run_controlled(25)
        with patch.object(plt, "show"):
            simulation.plots.start_live()
        try:
            plt.close(simulation.plots._figures["linear_velocity"])
            simulation.plots.refresh()
            with tempfile.TemporaryDirectory() as directory:
                self.assertEqual(len(simulation.plots.save(directory)), 4)
        finally:
            simulation.plots.close()


if __name__ == "__main__":
    unittest.main()

"""Simulation telemetry and Cartesian PBVS plots, using measured robot state.

The trajectory is the camera/tool path reconstructed from measured joint
positions. Velocities are J(q) @ dq, not finite differences of noisy poses.
Measured motion, PBVS reference, and applied command share world axes.
Matplotlib is imported only when displaying or exporting figures.
"""

import csv
from pathlib import Path

import numpy as np


class SimulationPlots:
    """Record control-rate samples; show live figures or save PNGs and CSV."""

    def __init__(self):
        self._samples = []
        self._figures = {}
        self._lines = {}
        self._live = False

    def reset(self):
        self._samples.clear()

    def record(self, state, kinematics, result):
        """Use an interface RobotState and a separate prediction workspace.

        reference_velocity is PBVS's joint reference during approach. Pickup
        has a different controller, so its PBVS reference is logged as NaN.
        Recording never accesses live simulator data or sends commands.
        """
        kinematics.synchronize(state)
        camera = kinematics.camera_pose()
        tool = kinematics.tool_pose()
        optical_jacobian = kinematics.camera_jacobian(state.joint_positions)
        world_jacobian = np.vstack((camera[:3, :3] @ optical_jacobian[:3],
                                   camera[:3, :3] @ optical_jacobian[3:]))
        measured_twist = world_jacobian @ state.joint_velocities
        commanded_twist = world_jacobian @ result.command_velocity
        reference_twist = (np.full(6, np.nan) if result.status.startswith("pickup_")
                           else world_jacobian @ result.reference_velocity)
        cube = np.full(3, np.nan)
        if result.pose_estimate is not None:
            cube = result.pose_estimate.world_from_cube[:3, 3]
        elif result.observation is not None:
            cube = result.observation.world_from_cube[:3, 3]
        sample = {
            "timestamp": float(state.timestamp), "status": result.status,
            "camera_position": camera[:3, 3].copy(),
            "tool_position": tool[:3, 3].copy(), "cube_position": cube.copy(),
            "linear_velocity": measured_twist[:3].copy(),
            "angular_velocity": measured_twist[3:].copy(),
            "reference_linear_velocity": reference_twist[:3].copy(),
            "reference_angular_velocity": reference_twist[3:].copy(),
            "command_linear_velocity": commanded_twist[:3].copy(),
            "command_angular_velocity": commanded_twist[3:].copy(),
        }
        if self._samples and sample["timestamp"] < self._samples[-1]["timestamp"]:
            raise ValueError("plot timestamps must be monotonic; reset after restarting the clock")
        if self._samples and sample["timestamp"] == self._samples[-1]["timestamp"]:
            self._samples[-1] = sample
        else:
            self._samples.append(sample)

    @property
    def data(self):
        """Return owned column arrays; time is seconds since the first sample."""
        if not self._samples:
            return {"time": np.empty(0)}
        values = {name: np.asarray([sample[name] for sample in self._samples])
                  for name in self._samples[0]}
        values["time"] = values["timestamp"] - values["timestamp"][0]
        return values

    def _create_figures(self, live=False):
        if self._figures:
            return
        try:
            from matplotlib.figure import Figure
        except ImportError as error:
            raise RuntimeError("Simulation plots require matplotlib; install requirements.txt") from error
        if live:
            from matplotlib import pyplot as plt
            make_figure = plt.figure
        else:
            # Export without pyplot or a GUI backend, including --headless.
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            def make_figure(**kwargs):
                figure = Figure(**kwargs)
                FigureCanvasAgg(figure)
                return figure
        self._live = live
        trajectory = make_figure(figsize=(8, 6), constrained_layout=True)
        axes = trajectory.add_subplot(111, projection="3d")
        for name, label, style in (
            ("camera_position", "Measured camera", "-"),
            ("tool_position", "Measured tool flange", "-"),
            ("cube_position", "Estimated cube", "--"),
        ):
            self._lines[name], = axes.plot([], [], [], style, label=label)
        axes.set(xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)",
                 title="PBVS / MPC trajectory")
        axes.legend()
        self._figures["trajectory"] = trajectory
        for quantity, unit in (("linear_velocity", "m/s"), ("angular_velocity", "rad/s")):
            figure = make_figure(figsize=(9, 7), constrained_layout=True)
            subplots = figure.subplots(3, 1, sharex=True)
            for index, axis in enumerate(subplots):
                for prefix, label, style in (
                    ("", "Measured", "-"), ("reference_", "PBVS reference", "--"),
                    ("command_", "Applied joint command", ":"),
                ):
                    key = prefix + quantity
                    self._lines[(key, index)], = axis.plot([], [], style, label=label)
                axis.set_ylabel(f"{'xyz'[index]} ({unit})")
                axis.grid(alpha=0.3)
            subplots[0].set_title(quantity.replace("_", " ").capitalize()
                                  + " — camera, world axes")
            subplots[0].legend(ncol=3, fontsize=8)
            subplots[-1].set_xlabel("Simulation time since first sample (s)")
            self._figures[quantity] = figure

    def refresh(self):
        """Update existing figures. The simulation throttles live redraws."""
        if not self._figures or not self._samples:
            return
        values = self.data
        for key in ("camera_position", "tool_position", "cube_position"):
            self._lines[key].set_data_3d(*values[key].T)
        axes = self._figures["trajectory"].axes[0]
        points = np.vstack([values[key] for key in ("camera_position", "tool_position", "cube_position")])
        points = points[np.all(np.isfinite(points), axis=1)]
        lower, upper = points.min(axis=0), points.max(axis=0)
        # Equal metric scale on all three axes avoids distorting the path.
        center = (lower + upper) / 2
        radius = max(float(np.max(upper - lower)) / 2 * 1.1, 0.02)
        axes.set_xlim(center[0] - radius, center[0] + radius)
        axes.set_ylim(center[1] - radius, center[1] + radius)
        axes.set_zlim(center[2] - radius, center[2] + radius)
        axes.set_box_aspect((1, 1, 1))
        for quantity in ("linear_velocity", "angular_velocity"):
            for index, axis in enumerate(self._figures[quantity].axes):
                for prefix in ("", "reference_", "command_"):
                    key = prefix + quantity
                    self._lines[(key, index)].set_data(values["time"], values[key][:, index])
                axis.relim()
                axis.autoscale_view()
        for figure in self._figures.values():
            if self._live:
                from matplotlib import pyplot as plt
                if not plt.fignum_exists(figure.number):
                    continue  # Closing a plot window does not stop the robot viewer.
            figure.canvas.draw_idle()
            if self._live:
                figure.canvas.flush_events()

    def start_live(self):
        """Open nonblocking plot windows, on the simulation's main thread."""
        if self._figures and not self._live:
            self.close()  # Replace export-only figures with managed GUI figures.
        self._create_figures(live=True)
        from matplotlib import pyplot as plt
        plt.show(block=False)

    def save(self, directory="simulation_plots"):
        """Save three PNG figures and trajectory.csv; return their paths."""
        if not self._samples:
            return []
        self._create_figures()
        self.refresh()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, figure in self._figures.items():
            path = directory / f"{name}.png"
            figure.savefig(path, dpi=160)
            paths.append(path)
        path = directory / "trajectory.csv"
        values = self.data
        names = [name for name in self._samples[0] if name not in ("timestamp", "status")]
        headers = [f"{name}_{component}" for name in names for component in "xyz"]
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["time_s", "timestamp_s", "status"] + headers)
            for index in range(len(self._samples)):
                row = [values["time"][index], values["timestamp"][index], values["status"][index]]
                writer.writerow(row + np.concatenate([values[name][index] for name in names]).tolist())
        paths.append(path)
        return paths

    def close(self):
        if self._live:
            from matplotlib import pyplot as plt
            for figure in self._figures.values():
                plt.close(figure)
        self._figures.clear()
        self._lines.clear()
        self._live = False

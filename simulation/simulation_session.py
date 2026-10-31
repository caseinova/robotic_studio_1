"""Simulation-clock pick/place using the same actions as the hardware session."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import mujoco
import numpy as np

from common.camera_processing import TaggedCubeProcessor
from common.controller import MPCConfig
from common.geometry import PlaneObstacle
from simulation.kinematics import MujocoKinematics
from common.placement import PlaceAction
from simulation.rg2_model import angles_to_width, linkage_positions
from common.pickup import ActionResult, PickAction


def add_destination_scene(simulation, destinations, positions):
    """Compile visual destination tags and a free cube before starting physics."""
    robot = simulation.robot
    spec = mujoco.MjSpec.from_file(str(robot.model_path))
    cube = spec.body('cube')
    cube.pos = simulation.config.cube_position
    cube.quat = simulation.config.cube_quaternion
    cube.add_freejoint(name='cube_free_joint')
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    with TemporaryDirectory(prefix='rg2-destination-') as directory:
        for index, (name, destination) in enumerate(destinations.items()):
            if name not in positions:
                continue
            image = np.full((360, 360), 255, dtype=np.uint8)
            image[20:340, 20:340] = cv2.aruco.generateImageMarker(dictionary, destination.marker_id, 320)
            path = Path(directory) / f'{index}.png'
            if not cv2.imwrite(str(path), image):
                raise RuntimeError('could not generate simulation destination marker')
            texture = f'destination_texture_{index}'
            material = f'destination_material_{index}'
            spec.add_texture(name=texture, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(path))
            mat = spec.add_material(name=material, texuniform=False, texrepeat=[1, 1], specular=0, shininess=0)
            mat.textures = ['', texture, '', '', '', '', '', '', '', '']
            body = spec.worldbody.add_body(name=f'destination_{name}', pos=positions[name])
            half = destination.marker_size_m * 360 / 320 / 2
            body.add_geom(name=f'destination_geom_{index}', type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=[0, 0, .00005], size=[half, half, .00005], material=material,
                          contype=0, conaffinity=0, mass=0)
        model = spec.compile()
    robot.model, robot.data = model, mujoco.MjData(model)
    robot.reset()


class SimulationPlacementKinematics(MujocoKinematics):
    """An isolated geometry workspace; predictions cannot teleport the real cube."""

    def __init__(self, robot, support_name):
        shadow = copy.copy(robot)
        shadow.model = copy.copy(robot.model)
        super().__init__(shadow)
        self.obstacles = (PlaneObstacle(support_name, [0, 0, 1], 0.),)

    @property
    def gripper_width(self):
        return angles_to_width(self.data.qpos[self.gripper_indices[[0, 3]]])

    @gripper_width.setter
    def gripper_width(self, width):
        self.data.qpos[self.gripper_indices] = linkage_positions(width)
        mujoco.mj_forward(self.model, self.data)

    def set_obstacles(self, obstacles):
        # This session has one physical cube. Its deposited pose is retained by
        # the guard; subsequent picks obtain its new pose from the camera.
        self.obstacles = tuple(obstacles)


class SimulationPlacementGuard:
    """MuJoCo mesh distances, including a carried cube or its release envelope."""

    def __init__(self, kinematics, size, clearance, tool_from_cube, support_name):
        self.kinematics = kinematics
        self.size, self.clearance = size, clearance
        self.tool_from_cube = tool_from_cube.copy()
        self.released_pose = None
        self.released_half_sizes = np.full(3, size / 2)
        self.cube_geom = kinematics.model.geom('cube_geom').id
        self.cube_body = kinematics.model.body('cube').id
        self.cube_qpos = kinematics.model.joint('cube_free_joint').qposadr[0]
        self.fingers = {kinematics.model.geom(f'gripper_{side}_geom').id for side in ('left', 'right')}
        self.floor = kinematics.model.geom('floor').id

    def check(self, q):
        k = self.kinematics
        k.set_positions(q)
        cube = self.released_pose if self.released_pose is not None else k.tool_pose() @ self.tool_from_cube
        address = self.cube_qpos
        k.data.qpos[address:address+3] = cube[:3, 3]
        mujoco.mju_mat2Quat(k.data.qpos[address+3:address+7], cube[:3, :3].flatten())
        k.model.geom_size[self.cube_geom] = (self.released_half_sizes if self.released_pose is not None
                                            else np.full(3, self.size / 2))
        mujoco.mj_forward(k.model, k.data)
        pairs = [(a, b, self.clearance) for a, b in k.pairs if self.cube_geom not in (a, b)]
        for geom in range(k.model.ngeom):
            if (geom == self.cube_geom or k.model.geom_bodyid[geom] == self.cube_body
                    or not (k.model.geom_contype[geom] or k.model.geom_conaffinity[geom])):
                continue
            if geom in self.fingers and self.released_pose is None:
                continue
            pairs.append((self.cube_geom, geom, 0. if geom == self.floor or geom in self.fingers
                          else self.clearance))
        for a, b, required in pairs:
            distance = mujoco.mj_geomDistance(k.model, k.data, a, b, max(required+.01, .05), np.zeros(6))
            if distance < required - 1e-9:
                raise RuntimeError(f'placement_path_blocked: {k.model.geom(a).name} / '
                                   f'{k.model.geom(b).name}; clearance={distance*1000:.1f} mm')


class SimulationSession:
    """Blocking composable actions. Physics advances while each action runs."""

    def __init__(self, simulation, sequence_config, *, on_update=None):
        self.simulation = simulation
        self.config = sequence_config
        self.controller_config = simulation.controller_config
        self.estimator_config = simulation.estimator_config
        self.interface = simulation.interface
        self.settings = MPCConfig()
        self.cube_size = simulation.robot.cube_size
        self.tool_from_cube = None
        self.status, self.diagnostic, self.tracking_report = 'idle', '', ''
        self.camera_view = on_update is not None
        self.latest_frame = None
        self.on_update = on_update
        self.failed = False
        self.kinematics = None

    def validate_destination(self, name):
        if name not in self.config.destinations:
            raise ValueError(f'unknown destination: {name}')
        if name not in self.simulation.config.destination_positions:
            raise ValueError(f'missing simulation destination_positions entry: {name}')
        destination = self.config.destinations[name]
        if destination.marker_id in range(7):
            raise ValueError('destination ID overlaps a cube/tool marker')
        return destination

    def __enter__(self):
        for name in self.config.destinations:
            self.validate_destination(name)
        add_destination_scene(self.simulation, self.config.destinations,
                              self.simulation.config.destination_positions)
        self.kinematics = MujocoKinematics(self.simulation.robot)
        return self

    def __exit__(self, *_):
        self.interface.set_joint_velocities(np.zeros(6))
        self.interface.close()
        self.simulation.plots.close()

    def publish_preview(self, frame):
        self.latest_frame = frame

    def _run(self, action, name):
        if self.failed:
            raise RuntimeError('simulation session failed; restart before another action')
        started = self.interface.timestamp
        try:
            while True:
                if self.interface.timestamp - started > self.config.action_timeout_s:
                    raise TimeoutError(f'{name} action_timeout')
                done = action.update()
                if self.on_update is not None:
                    self.on_update(self)
                if done:
                    self.status = name + '_complete'
                    return ActionResult(name, 'complete', self.tool_from_cube is not None)
                until = self.interface.timestamp + self.settings.dt
                while self.interface.timestamp < until - 1e-9:
                    self.simulation.step()
        except BaseException:
            self.failed = True
            raise
        finally:
            self.interface.set_joint_velocities(np.zeros(6))

    def servo_to_pick_up(self):
        if self.kinematics is None:
            raise RuntimeError('enter the simulation session first')
        if self.tool_from_cube is not None:
            raise RuntimeError('already holding a cube')
        # Fresh workspace uses the current visual target, never physics truth.
        self.kinematics = MujocoKinematics(self.simulation.robot)
        processor = TaggedCubeProcessor(cube_size=self.cube_size,
                                        marker_size=self.simulation.robot.cube_marker_size)
        return self._run(PickAction(self, processor=processor), 'pick')

    def move_to_destination_and_drop(self, name):
        destination = self.validate_destination(name)
        if self.tool_from_cube is None:
            raise RuntimeError('pick up a cube before placement')
        self.kinematics = SimulationPlacementKinematics(self.simulation.robot,
                                                       destination.collision_support_plane)
        action = PlaceAction(self, destination, guard_factory=SimulationPlacementGuard,
            frame_max_age=self.settings.dt*self.controller_config.frame_max_age_intervals,
            acceleration_limit=self.settings.acceleration_limit)
        return self._run(action, 'place')


def run_simulation_sequence(runtime, sequence_config, destination, *, validate_only=False):
    from contextlib import ExitStack
    import time
    from common.controller_config import ControllerConfig
    from common.estimator import EKFConfig
    from simulation.noise_config import NoiseConfig
    from simulation.simulation_config import SimulationConfig
    from simulation.simulate import Simulation

    experiment = SimulationConfig.load(runtime.simulation_config)
    controller = ControllerConfig.load(runtime.controller_config)
    estimator = EKFConfig.load(runtime.estimator_config)
    noise = NoiseConfig.load(runtime.noise_config)
    if destination not in sequence_config.destinations:
        raise ValueError(f'unknown destination: {destination}')
    for name, tag in sequence_config.destinations.items():
        if name not in experiment.destination_positions:
            raise ValueError(f'missing simulation destination_positions entry: {name}')
        if tag.marker_id in range(7):
            raise ValueError('destination ID overlaps a cube/tool marker')
    if validate_only:
        print('Simulation sequence configuration valid; no devices or simulation opened.')
        return 0
    simulation = Simulation(config=experiment, controller_config=controller,
                            estimator_config=estimator, noise_config=noise)
    with ExitStack() as stack:
        session = stack.enter_context(SimulationSession(simulation, sequence_config))
        viewer = None
        if not runtime.headless:
            import mujoco.viewer
            viewer = stack.enter_context(mujoco.viewer.launch_passive(simulation.robot.model, simulation.robot.data))
        camera_view = runtime.camera and not runtime.headless
        if camera_view:
            cv2.namedWindow('Simulation pick and place', cv2.WINDOW_NORMAL)
            stack.callback(cv2.destroyWindow, 'Simulation pick and place')
        last = [None, -np.inf, time.monotonic()]

        def update(s):
            if s.status != last[0] or s.interface.timestamp-last[1] >= 1.:
                print(f'Simulation: {s.status} | {s.diagnostic}', flush=True)
                last[:2] = [s.status, s.interface.timestamp]
            if viewer is not None:
                if not viewer.is_running():
                    raise KeyboardInterrupt
                viewer.sync()
            if camera_view:
                frame = s.latest_frame
                if frame is None or frame.timestamp != s.interface.timestamp:
                    frame = s.interface.capture_rgbd()
                cv2.imshow('Simulation pick and place', cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xff in (27, ord('q'), ord('Q')):
                    raise KeyboardInterrupt
            if viewer is not None:
                time.sleep(max(0., s.settings.dt - (time.monotonic()-last[2])))
                last[2] = time.monotonic()

        session.on_update = update
        session.camera_view = camera_view
        print(session.servo_to_pick_up(), flush=True)
        print(session.move_to_destination_and_drop(destination), flush=True)
        # Allow gravity/contact settling after retreat, without attaching the cube.
        simulation.step(int(.5/simulation.robot.model.opt.timestep))
        update(session)
    return 0

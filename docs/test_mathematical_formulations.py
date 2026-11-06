"""Numerical checks of the page's conventions against implemented operations.

These verify local derivatives and identities, not global controller stability
or the physical accuracy of the chosen stochastic models.
"""
import unittest

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from scipy.stats import chi2, norm

from common.camera_processing import CameraProcessor
from common.controller import PBVS
from common.estimator import EKFConfig, RelativePoseEKF, _left_jacobian, _skew
from common.tool_increment_ekf import ToolIncrementEKF
from hardware.hardware_kinematics import UR3Kinematics


def pose(position, rotvec):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    result[:3, 3] = position
    return result


def rotation_error(actual, nominal):
    return Rotation.from_matrix(actual @ nominal.T).as_rotvec()


def derivative(function, size):
    step = 1e-6
    return np.column_stack([
        (function(step * axis) - function(-step * axis)) / (2 * step)
        for axis in np.eye(size)
    ])


class MathematicalFormulationTests(unittest.TestCase):
    def test_world_and_tool_jacobians_match_fk_derivatives(self):
        kinematics = UR3Kinematics(
            joint_limits=(-7 * np.ones(6), 7 * np.ones(6)),
            world_from_base=pose([.2, -.1, .3], [.1, -.3, .2]),
            flange_from_tool=pose([.01, .02, .08], [.1, .2, -.2]),
            tool_from_grasp=np.eye(4))
        for q in ([.3, -.7, .8, -.4, .9, .2], [-1., -1.5, 1.7, -1.8, -1.6, -.2]):
            q = np.asarray(q)
            nominal = kinematics.forward_kinematics(q)

            def error(offset):
                moved = kinematics.forward_kinematics(q + offset)
                return np.r_[moved[:3, 3] - nominal[:3, 3],
                             rotation_error(moved[:3, :3], nominal[:3, :3])]

            jacobian = kinematics.tool_jacobian(q)
            np.testing.assert_allclose(jacobian, derivative(error, 6), atol=1e-9)
            rotate = np.kron(np.eye(2), nominal[:3, :3].T)
            np.testing.assert_allclose(kinematics.tool_jacobian(q, local=True),
                                       rotate @ jacobian, atol=1e-12)

    def test_common_camera_transform_cancels_before_filtering(self):
        camera = pose([.3, -.8, .5], [.4, -.2, 1.])
        tool = pose([.1, .1, .8], [.2, -.7, .4])
        cube = pose([-.1, .2, .7], [-.6, .1, .3])
        np.testing.assert_allclose(np.linalg.inv(camera @ tool) @ (camera @ cube),
                                   np.linalg.inv(tool) @ cube, atol=1e-12)

    def test_ray_plane_reconstruction_matches_implemented_fit(self):
        intrinsics = np.array([[600., 0., 320.], [0., 610., 240.], [0., 0., 1.]])
        pixels = np.array([[x, y, 1.] for x in np.linspace(250, 390, 6)
                           for y in np.linspace(180, 300, 6)])
        rays = pixels @ np.linalg.inv(intrinsics).T
        normal = np.array([.2, -.4, 1.])
        normal /= np.linalg.norm(normal)
        points = rays * (.7 / (rays @ normal))[:, None]
        fitted_normal, distance = CameraProcessor._fit_depth_plane(points)
        reconstructed = rays * (distance / (rays @ fitted_normal))[:, None]
        np.testing.assert_allclose(reconstructed, points, atol=1e-12)
        projection = points @ intrinsics.T
        np.testing.assert_allclose(projection[:, :2] / projection[:, 2, None],
                                   pixels[:, :2], atol=1e-12)

    def test_cube_transition_matches_left_error_finite_difference(self):
        p, v, omega = np.array([.1, .2, .3]), np.array([.2, -.1, .05]), np.array([.7, -.5, .9])
        rotation = Rotation.from_rotvec([.3, -.6, .2]).as_matrix()
        h = .04
        advance = Rotation.from_rotvec(h * omega).as_matrix()
        expected = np.eye(12)
        expected[:3, 6:9] = h * np.eye(3)
        expected[3:6, 3:6] = advance
        expected[3:6, 9:12] = h * _left_jacobian(h * omega)

        def propagate_error(error):
            perturbed_rotation = Rotation.from_rotvec(error[3:6]).as_matrix() @ rotation
            moved_rotation = Rotation.from_rotvec(h * (omega + error[9:12])).as_matrix() @ perturbed_rotation
            return np.r_[p + error[:3] + h * (v + error[6:9]) - (p + h * v),
                         rotation_error(moved_rotation, advance @ rotation),
                         error[6:9], error[9:12]]

        np.testing.assert_allclose(expected, derivative(propagate_error, 12), atol=1e-9)
        config = EKFConfig()
        ekf = RelativePoseEKF(config)
        ekf.update(0., pose(p, [.3, -.6, .2]), np.eye(4))
        ekf._velocity, ekf._angular_velocity = v.copy(), omega.copy()
        before = ekf._covariance.copy()
        noise = np.zeros((12, 12))
        for start, velocity, sigma in ((0, 6, config.linear_acceleration_noise),
                                       (3, 9, config.angular_acceleration_noise)):
            for axis in range(3):
                indices = [start + axis, velocity + axis]
                noise[np.ix_(indices, indices)] = sigma**2 * np.array(
                    [[h**3 / 3, h**2 / 2], [h**2 / 2, h]])
        ekf._propagate(h)
        np.testing.assert_allclose(ekf._covariance, expected @ before @ expected.T + noise,
                                   atol=1e-12)

    def test_tool_increment_transition_and_covariance(self):
        initial = pose([.1, -.2, .4], [.5, -.3, .2])
        increment = pose([.02, -.03, .01], [.2, -.4, .1])
        nominal = initial @ increment
        expected = np.eye(6)
        expected[:3, 3:] = -_skew(initial[:3, :3] @ increment[:3, 3])

        def propagate_error(error):
            perturbed = initial.copy()
            perturbed[:3, 3] += error[:3]
            perturbed[:3, :3] = Rotation.from_rotvec(error[3:]).as_matrix() @ initial[:3, :3]
            moved = perturbed @ increment
            return np.r_[moved[:3, 3] - nominal[:3, 3],
                         rotation_error(moved[:3, :3], nominal[:3, :3])]

        np.testing.assert_allclose(expected, derivative(propagate_error, 6), atol=1e-9)
        config = EKFConfig(prediction_model='fk_increment')
        ekf = ToolIncrementEKF(config)
        ekf.step(0., initial)
        before = ekf.covariance.copy()
        ekf.step(.05, nominal)
        noise = .05 * np.diag([config.fk_position_noise**2] * 3
                              + [config.fk_orientation_noise**2] * 3)
        np.testing.assert_allclose(ekf.pose, nominal, atol=1e-12)
        np.testing.assert_allclose(ekf.covariance, expected @ before @ expected.T + noise,
                                   atol=1e-12)

    def test_reset_uses_positive_left_jacobian(self):
        for correction in (np.zeros(3), np.array([.3, -.2, .4])):
            nominal = Rotation.from_rotvec(correction).as_matrix()
            numerical = derivative(lambda error: rotation_error(
                Rotation.from_rotvec(correction + error).as_matrix(), nominal), 3)
            np.testing.assert_allclose(_left_jacobian(correction), numerical, atol=1e-9)

    def test_rotational_noise_block_is_only_exact_at_zero_rate(self):
        h = .05
        simple = np.kron(np.array([[h**3 / 3, h**2 / 2], [h**2 / 2, h]]), np.eye(3))
        diffusion = np.diag([0.] * 3 + [1.] * 3)
        for omega in (np.zeros(3), np.array([2., -1., .5])):
            dynamics = np.zeros((6, 6))
            dynamics[:3, :3] = _skew(omega)
            dynamics[:3, 3:] = np.eye(3)
            exact, _ = quad_vec(lambda t: expm(dynamics * t) @ diffusion
                                @ expm(dynamics.T * t), 0., h)
            if np.linalg.norm(omega) == 0:
                np.testing.assert_allclose(exact, simple, atol=1e-12)
            else:
                self.assertGreater(np.linalg.norm(exact - simple), 1e-5)

    def test_pbvs_error_sign_limits_and_damped_reference(self):
        desired = pose([.03, -.02, .1], [.3, .1, -.2])
        error_transform = pose([.001, -.002, .003], [.002, .001, -.003])
        singular_values = np.array([1., .8, .5, .2, .05, .01])
        jacobian = np.diag(singular_values)
        controller = PBVS(desired)
        command, error = controller.reference(error_transform @ desired, jacobian)
        expected_error = np.r_[error_transform[:3, 3], [.002, .001, -.003]]
        np.testing.assert_allclose(error, expected_error, atol=1e-9)
        expected_command = (singular_values / (singular_values**2 + controller.damping**2)
                            * controller.gain * error)
        np.testing.assert_allclose(command, expected_command, atol=1e-10)
        rates = controller.gain * singular_values**2 / (singular_values**2 + controller.damping**2)
        self.assertTrue(np.all((rates > 0) & (rates < controller.gain)))
        large = pose([.2, -.3, .1], [.4, .5, -.2])
        command, _ = controller.reference(large @ desired, np.eye(6))
        twist = command * (1 + controller.damping**2)
        self.assertAlmostEqual(np.linalg.norm(twist[:3]), controller.linear_speed)
        self.assertAlmostEqual(np.linalg.norm(twist[3:]), controller.angular_speed)

    def test_nominal_gaussian_interval_and_ellipsoid_coverage_differ(self):
        self.assertAlmostEqual(norm.cdf(2) - norm.cdf(-2), .9544997361)
        self.assertAlmostEqual(chi2.cdf(4, df=3), .7385358701)


if __name__ == '__main__':
    unittest.main()

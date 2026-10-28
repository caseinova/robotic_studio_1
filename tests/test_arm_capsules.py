import numpy as np

from hardware_config import HardwareConfig
from tools.derive_arm_capsules import derive, fit, fit_parts, coarse_parts, partition_triangles, segment_distances


def test_zero_length_capsule_distance():
    np.testing.assert_allclose(segment_distances(np.eye(3), np.zeros(3), np.zeros(3)), 1)


def test_fit_preserves_margin_on_short_wide_shape():
    angles = np.linspace(0, 2*np.pi, 40, endpoint=False)
    points = np.array([[.07*np.cos(a), .07*np.sin(a), z]
                       for a in angles for z in (0., .02)])
    start, end, radius = fit(points)
    assert np.all(segment_distances(points, start, end) <= radius-.003+1e-10)
    assert radius < .08


def test_parts_enclose_whole_triangles_not_just_nearest_vertices():
    triangles = np.random.default_rng(3).normal(size=(60, 3, 3)) * .03
    parts = fit_parts(triangles.reshape(-1, 3))
    def area(triangles):
        return np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                       triangles[:, 2]-triangles[:, 0]), axis=1).sum()/2
    np.testing.assert_allclose(sum(area(group.reshape(-1, 3, 3)) for group, _ in parts), area(triangles))
    for group, (start, end, radius) in parts:
        assert np.all(segment_distances(group, start, end) <= radius-.003+1e-10)
        centers = group.reshape(-1, 3, 3).mean(axis=1)
        assert np.all(segment_distances(centers, start, end) <= radius-.003+1e-10)


def test_subdivision_preserves_boundary_crossing_triangle_area():
    triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    parts = partition_triangles(triangle, np.array([1., 0., 0.]), 4)
    faces = np.concatenate(parts).reshape(-1, 3, 3)
    area = np.linalg.norm(np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0]), axis=1).sum()/2
    np.testing.assert_allclose(area, .5)


def test_refinement_stays_inside_coarse_capsules():
    points = np.random.default_rng(9).normal(size=(60, 3, 3)).reshape(-1, 3)*.03
    parents = [c for _, c in coarse_parts(points)]
    for _, (a, b, radius) in fit_parts(points):
        assert any(np.max(segment_distances(np.array([a, b]), x, y))+radius <= r+1e-10
                   for x, y, r in parents)


def test_bundled_mesh_derivation_and_named_collision_pairs():
    config = HardwareConfig.load()
    capsules, report = derive(config)
    assert len({c['name'] for c in capsules}) == len(capsules)
    assert {c['frame'] for c in capsules} == set(range(7))
    assert all(link['minimum_triangle_margin_m'] >= .003-1e-10 for link in report)
    model = config.build_kinematics()
    for index in model.pickup_collision_indices():
        assert ' / ' in model.collision_pair_name(index)

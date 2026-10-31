"""Backend-independent collision shapes and rigid-transform validation."""
from dataclasses import dataclass
import numpy as np
from common.kinematics import inverse_pose

def _vector(value, size, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    result.setflags(write=False)
    return result


def _pose(value, name):
    result = np.array(value, dtype=float, copy=True)
    if (result.shape != (4, 4) or not np.all(np.isfinite(result))
            or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(result[:3, :3]), 1, atol=1e-8)):
        raise ValueError(f"{name} must be a rigid 4x4 transform")
    result.setflags(write=False)
    return result


def _positive(value, name):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class LinkCapsule:
    """Physical link approximation in DH frame 0..6 or the tool frame.

    frame 0 is base, frame i follows DH joint i, and 'tool' includes the
    flange-to-tool calibration. opening_axis shifts both endpoints by
    opening_axis * measured gripper width (e.g. (0, .5, 0) for one finger).
    contact_group='finger' permits contact only with 'grasp_target' obstacles
    in the existing pickup sequence. Adjacent-link self pairs are skipped.
    """

    name: str
    frame: int | str
    start: tuple
    end: tuple
    radius: float
    opening_axis: tuple = (0, 0, 0)
    contact_group: str = ""

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("capsule needs a name")
        if not (self.frame == "tool" or (type(self.frame) is int and 0 <= self.frame <= 6)):
            raise ValueError("capsule frame must be a DH frame 0..6 or 'tool'")
        for name in ("start", "end", "opening_axis"):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name))
        _positive(self.radius, "capsule radius")


@dataclass(frozen=True)
class SphereObstacle:
    name: str
    center: tuple
    radius: float
    contact_group: str = ""
    exclude_links: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "center", _vector(self.center, 3, "sphere center"))
        _positive(self.radius, "sphere radius")


@dataclass(frozen=True)
class BoxObstacle:
    """Oriented box; half_sizes in its own axes, pose in the world frame."""

    name: str
    world_from_box: np.ndarray
    half_sizes: tuple
    contact_group: str = ""
    exclude_links: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "world_from_box", _pose(self.world_from_box, "world_from_box"))
        half = _vector(self.half_sizes, 3, "box half_sizes")
        if np.any(half <= 0):
            raise ValueError("box half_sizes must be positive")
        object.__setattr__(self, "half_sizes", half)


@dataclass(frozen=True)
class PlaneObstacle:
    """Allowed half-space normal.dot(point) >= offset; normal is unit length."""

    name: str
    normal: tuple
    offset: float
    contact_group: str = ""
    exclude_links: tuple = ()

    def __post_init__(self):
        normal = _vector(self.normal, 3, "plane normal")
        if not np.isclose(np.linalg.norm(normal), 1) or not np.isfinite(self.offset):
            raise ValueError("plane needs a unit normal and finite offset")
        object.__setattr__(self, "normal", normal)


def _segment_distance(a, b, c, d):
    """Minimum Euclidean distance, including parallel and zero-length segments."""
    u, v, w = b - a, d - c, a - c
    aa, bb, cc = u @ u, u @ v, v @ v
    dd, ee = u @ w, v @ w
    candidates = []
    for s in (0., 1.):
        t = np.clip((ee + s * bb) / cc, 0, 1) if cc > 1e-20 else 0.
        candidates.append(np.linalg.norm(w + s * u - t * v))
    for t in (0., 1.):
        s = np.clip((t * bb - dd) / aa, 0, 1) if aa > 1e-20 else 0.
        candidates.append(np.linalg.norm(w + s * u - t * v))
    determinant = aa * cc - bb * bb
    if determinant > 1e-12 * aa * cc:
        s, t = (bb * ee - cc * dd) / determinant, (aa * ee - bb * dd) / determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append(np.linalg.norm(w + s * u - t * v))
    return float(min(candidates))


def _segment_distances(a, b, c, d):
    """Batched equivalent of _segment_distance, preserving degenerate cases."""
    u, v, w = b - a, d - c, a - c
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    aa, bb, cc = dot(u, u), dot(u, v), dot(v, v)
    dd, ee = dot(u, w), dot(v, w)
    candidates = []
    for s in (0., 1.):
        t = np.clip(np.divide(ee + s * bb, cc, out=np.zeros_like(cc), where=cc > 1e-20), 0, 1)
        candidates.append(np.linalg.norm(w + s * u - t[:, None] * v, axis=1))
    for t in (0., 1.):
        s = np.clip(np.divide(t * bb - dd, aa, out=np.zeros_like(aa), where=aa > 1e-20), 0, 1)
        candidates.append(np.linalg.norm(w + s[:, None] * u - t * v, axis=1))
    determinant = aa * cc - bb * bb
    valid = determinant > 1e-12 * aa * cc
    s = np.divide(bb * ee - cc * dd, determinant, out=np.zeros_like(aa), where=valid)
    t = np.divide(aa * ee - bb * dd, determinant, out=np.zeros_like(aa), where=valid)
    interior = np.linalg.norm(w + s[:, None] * u - t[:, None] * v, axis=1)
    candidates.append(np.where(valid & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1), interior, np.inf))
    return np.min(candidates, axis=0)


def _segment_box_distance(start, end, half):
    """Exact segment-to-solid-AABB distance, zero when intersecting.

    Distance squared is piecewise quadratic. Partition at face crossings and
    check each interval's stationary point, so thin boxes cannot be skipped
    by sampling. Rotation into box coordinates is done by the caller.
    """
    direction = end - start
    breaks = [0., 1.]
    for axis in range(3):
        if abs(direction[axis]) > 1e-15:
            for face in (-half[axis], half[axis]):
                t = (face - start[axis]) / direction[axis]
                if 0 < t < 1:
                    breaks.append(float(t))
    breaks = sorted(set(breaks))
    candidates = list(breaks)
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        mid = start + (lo + hi) / 2 * direction
        outside = np.abs(mid) > half
        if not np.any(outside):
            return 0.
        boundary = np.sign(mid[outside]) * half[outside]
        delta, slope = start[outside] - boundary, direction[outside]
        if slope @ slope > 1e-20:
            candidates.append(float(np.clip(-(delta @ slope) / (slope @ slope), lo, hi)))
    return float(min(np.linalg.norm(np.maximum(np.abs(start + t * direction) - half, 0))
                     for t in candidates))



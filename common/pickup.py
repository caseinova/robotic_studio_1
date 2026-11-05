"""Backend-independent pick action and action results."""
from dataclasses import dataclass, replace
import numpy as np
from common.controller import ApproachController, CubePickupController, start_pickup_or_reapproach

def _pickup_target(result):
    observation = result.observation
    if result.pose_estimate is None:
        return observation
    import numpy as np

    estimate = result.pose_estimate
    cube_from_reference = np.linalg.inv(observation.camera_from_cube) @ observation.camera_from_marker
    return replace(observation, camera_from_cube=estimate.camera_from_cube.copy(),
        camera_from_marker=estimate.camera_from_cube @ cube_from_reference,
        world_from_cube=estimate.world_from_cube.copy())


@dataclass(frozen=True)
class ActionResult:
    action: str
    status: str
    holding_cube: bool


class PickAction:
    """Reuse approach -> opening -> dwell -> descend -> close -> lift."""

    def __init__(self, session, *, processor=None):
        self.session = session
        self._last_update = None
        self._last_source = None
        self._source_switches = 0
        settings, c = session.settings, session.controller_config
        self.approach = ApproachController(session.interface, session.kinematics,
            config=settings, controller_config=c, estimator_config=session.estimator_config,
            processor=session.hardware.build_processor() if processor is None else processor)
        self.pickup = CubePickupController(session.interface, session.kinematics,
            speed=c.pickup_linear_speed_m_s, angular_speed=c.pickup_angular_speed_rad_s,
            cube_size=session.cube_size, control_dt=settings.dt,
            velocity_limit=settings.velocity_limit,
            clearance=getattr(session, 'pickup_clearance', settings.clearance),
            frame_max_age_intervals=c.frame_max_age_intervals,
            grasp_height_offset_m=c.grasp_height_offset_m,
            pickup_dwell_seconds=c.pickup_dwell_seconds,
            pickup_dwell_radius_m=c.pickup_dwell_radius_m)

    def update(self):
        s = self.session
        if self.pickup.phase == "idle":
            s.interface.require_camera_for_motion = True
            started = s.interface.timestamp
            frame = s.interface.capture_rgbd()
            captured = s.interface.timestamp
            result = self.approach.update(frame)
            finished = s.interface.timestamp
            if getattr(s, 'camera_view', False) and getattr(s, 'annotations', True):
                rgb = self.approach.processor.annotate(frame.rgb, result.observation, frame.intrinsics)
                s.publish_preview(replace(frame, rgb=rgb))
            else:
                s.publish_preview(frame)
            source = result.tool_pose_source or "unavailable"
            if source in ("tag", "fk", "fk_fallback"):
                if self._last_source is not None and source != self._last_source:
                    self._source_switches += 1
                self._last_source = source
            interval = None if self._last_update is None else started - self._last_update
            self._last_update = started
            s.tracking_report = (
                f"status={result.status}, reason={result.diagnostic or 'none'}, "
                f"tool={source}, source_switches={self._source_switches}, "
                f"capture_ms={(captured-started)*1000:.1f}, "
                f"processing_ms={(finished-captured)*1000:.1f}, "
                f"frame_age_ms={(finished-frame.timestamp)*1000:.1f}, "
                f"frame_limit_ms={s.controller_config.frame_max_age_intervals*s.settings.dt*1000:.1f}, "
                f"interval_ms={'n/a' if interval is None else format(interval*1000, '.1f')}, "
                f"position_error_mm={result.position_error*1000:.1f}, "
                f"rotation_error_deg={np.degrees(result.rotation_error):.1f}")
            if result.diagnostic == "tag_metric_size_inconsistent":
                processor = self.approach.processor
                failures = getattr(processor, "metric_size_rejections", ())
                s.tracking_report += (
                    f", expected_tag_mm={processor.marker_size*1000:.1f}, tolerance=10%, "
                    "rejected_edges_mm=" + str([
                        (marker_id, np.round(edges*1000, 1).tolist()) for marker_id, edges in failures]))
            s.status = result.status
            s.diagnostic = result.diagnostic
            if result.status != "reached":
                return False
            s.interface.require_camera_for_motion = False
            result = start_pickup_or_reapproach(self.approach, self.pickup, _pickup_target(result))
        else:
            result = self.pickup.update()
        s.tracking_report = ""
        s.status = result.status
        s.diagnostic = result.diagnostic
        if self.pickup.phase == "failed":
            raise RuntimeError("pickup failed: " + result.diagnostic)
        if self.pickup.phase == "complete":
            s.tool_from_cube = self.pickup.tool_from_cube.copy()
            return True
        return False



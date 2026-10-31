from abc import ABC, abstractmethod


class RobotBackend(ABC):
    """
    Common interface used by both:

        MuJoCo simulation
        Real UR3e hardware

    Student control code should depend on this interface,
    not directly on MuJoCo or RTDE.
    """

    @abstractmethod
    def get_joint_positions(self):
        """
        Return the six UR3e joint positions [rad].

        Order:
            0 shoulder_pan
            1 shoulder_lift
            2 elbow
            3 wrist_1
            4 wrist_2
            5 wrist_3
        """
        pass

    @abstractmethod
    def get_joint_velocities(self):
        """
        Return the six UR3e joint velocities [rad/s].
        """
        pass

    @abstractmethod
    def set_joint_velocity(self, qd):
        """
        Send six joint velocity commands [rad/s].
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop commanded joint motion.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close communication / simulation resources.
        """
        pass
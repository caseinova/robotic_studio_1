import numpy as np

from roboticstoolbox.models.URDF.URDFRobot import URDFRobot


# ============================================================
# Load UR3e
#
# Robotics Toolbox -> robot_descriptions -> Universal Robots
# official UR3e Xacro
# ============================================================

robot = URDFRobot("ur3e")


# ============================================================
# Basic information
# ============================================================

print()
print("========================================")
print("UR3e Robotics Toolbox Model")
print("========================================")
print()

print(robot)

print()
print("Number of joints:")
print(robot.n)


# ============================================================
# Test configuration
# ============================================================

q = np.zeros(6)

print()
print("Joint configuration q [rad]:")
print(q)


# ============================================================
# Forward kinematics
# ============================================================

T = robot.fkine(
    q,
    end="tool0",
)
print()
print("Forward kinematics:")
print(T)


# ============================================================
# Jacobian
# ============================================================

J = robot.jacob0(
    q,
    end="tool0",
)
print()
print("Jacobian:")
print(J)

print()
print("Jacobian shape:")
print(J.shape)


# ============================================================
# Validation
# ============================================================

assert robot.n == 6

assert J.shape == (6, 6)


print()
print("========================================")
print("UR3e MODEL TEST PASSED")
print("========================================")
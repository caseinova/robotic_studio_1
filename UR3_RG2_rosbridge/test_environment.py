import sys

import numpy
import mujoco
import roboticstoolbox as rtb

import rtde_control
import rtde_receive


print("================================")
print("Python environment")
print("================================")

print("Python executable:")
print(sys.executable)

print()

print("Python version:")
print(sys.version)

print()

print("NumPy:")
print(numpy.__version__)

print()

print("MuJoCo:")
print(mujoco.__version__)

print()

print("Robotics Toolbox: OK")

print()

print("ur_rtde: OK")
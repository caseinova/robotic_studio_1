"""Simulation components for robotic studio."""
import os
import sys

# Select before importing MuJoCo/GLFW. Native Wayland prohibits window positioning.
# An explicit user choice always wins; headless rendering is unaffected.
if sys.platform.startswith('linux') and os.environ.get('DISPLAY'):
    os.environ.setdefault('PYGLFW_LIBRARY_VARIANT', 'x11')

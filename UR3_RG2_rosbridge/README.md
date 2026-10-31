# UR3_RG2_rosbridge

## Overview

1. Start with [Software setup](#software-setup). Check the installation with [test_environment.py](test_environment.py).
2. Try the [simulation examples](#simulation-examples). Check the model with [Example 03](examples/03_check_velocity_model.py). Test movement with [Example 04](examples/04_joint_velocity_sim.py).
3. Follow [Hardware setup](#hardware-setup) in the Mechatronics Lab. Borrow the device, connect it and prepare the arm.
4. Continue to the [robot examples](#robot-examples). Read the robot state with [Example 09](examples/09_real_ur_read_only.py). Test wrist movement with [Example 12](examples/12_real_speedj_minimal_test.py).

For RealSense camera work, follow [Working with RealSense](#working-with-realsense).

The software setup is only needed once. For later sessions, use [Returning to the lab](#returning-to-the-lab).

## Software setup

The current lab setup uses Ubuntu and Python 3.12. The instructions below also cover Windows and macOS, although these have not been tested with this project. The Python libraries and robot workflow are shared across all three systems.

Open a terminal. On Windows, use Command Prompt (`cmd`) for the commands below. Install the system tools first, then download the project.

### Install the system tools

Python runs the scripts. The pip tool installs Python libraries, and Git downloads the robot descriptions. Follow the instructions for the laptop's operating system.

#### Linux

On Ubuntu 24.04, the following commands install Python 3.12, pip, venv support and Git. The first command refreshes the package list. Installation may ask for the laptop password.

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git
```

For other Linux distributions, install Python 3.12, pip, venv support and Git through the distribution's package manager. Check that the required Python version is available before continuing.

```bash
python3.12 --version
```

#### Windows

Install Python 3.12 for x64 Windows with pip and the Python launcher. Install Git for Windows and make it available from the command line. Open a new Command Prompt after installation and check both tools.

```bat
py -3.12 --version
git --version
```

#### macOS

Install Python 3.12 with pip and install Git. Also have Xcode Command Line Tools, CMake and Boost installed. The `ur_rtde` package may need to compile during installation, particularly on Macs with Apple Silicon. See the [RTDE installation guide](https://sdurobotics.gitlab.io/ur_rtde/pages/getting_started/installation.html) for build requirements.

Open a new terminal after installation and check both tools.

```bash
python3.12 --version
git --version
```

### Download the project

After installing the tools, open a terminal in the folder where the project should be stored. If the project is already cloned, open its existing folder and continue to [Understand the Python environment](#understand-the-python-environment).

```text
git clone https://github.com/UTS-RI/UR3_RG2_rosbridge.git
cd UR3_RG2_rosbridge
```

The first command downloads the project and the Git information needed to switch branches. The second enters the project folder containing this README. Keep this terminal open and run the remaining commands from this folder.

### Understand the Python environment

A [virtual environment](https://docs.python.org/3.12/library/venv.html) is a folder with its own Python environment and installed libraries. It keeps this project's library versions separate from other projects and the operating system. For example, updating NumPy for another project can leave this project's version unchanged.

Using venv is optional for this project and recommended for a new setup. Choose one installation method below. Complete that method, then [check the installation](#check-the-installation).

Both methods install the same libraries.

* NumPy and SciPy handle numerical calculations and rotations.
* Robotics Toolbox provides robot kinematics and Jacobian calculations.
* MuJoCo provides the simulation and its viewer.
* ur_rtde communicates with the physical robot.
* robot_descriptions and xacrodoc load and convert robot models.

Installation needs internet access and may take a few minutes. The first robot model download also needs internet access. Using `python -m pip` selects pip from the same Python environment that runs the scripts.

### Recommended installation using venv

Create an environment named `ur3` inside the project folder, then activate it. Run only the commands for the laptop's operating system. Create the environment once and activate it each time a new terminal is opened.

An existing environment can be reused on the same computer. Create a fresh environment when changing computers or operating systems.

#### Linux and macOS

```bash
python3.12 -m venv ur3
source ur3/bin/activate
```

#### Windows Command Prompt

```bat
py -3.12 -m venv ur3
ur3\Scripts\activate.bat
```

#### Install the Python libraries

The prompt normally starts with `(ur3)`. Commands using `python` now use this environment, and pip installs libraries into its folder. The following commands are the same on all three operating systems.

Update pip so that it can recognise available package versions. Then install the project libraries.

```text
python -m pip install --upgrade pip
python -m pip install numpy scipy mujoco roboticstoolbox-python ur_rtde robot_descriptions "xacrodoc[mujoco]"
python -m pip install "git+https://github.com/RianPolMacCionaoith/onRobot.git@main#subdirectory=onRobot"
```

Continue to the [installation check](#check-the-installation).

<details>
<summary>Alternative installation without venv</summary>

Use this method only if the existing Python installation supports pip installations for the user account. Skip the venv commands above. Libraries installed this way are shared by projects using that account and Python installation.

Some Python installations, including Ubuntu's system Python, block this with an externally managed environment error. If that message appears, use the recommended venv method or a Python environment provided by the lab. [Ubuntu's Python guide](https://ubuntu.com/developers/docs/tutorials/python-use/) explains this restriction.

On Linux and macOS, install the libraries for the current user account with the following command.

```bash
python3.12 -m pip install --user numpy scipy mujoco roboticstoolbox-python ur_rtde robot_descriptions "xacrodoc[mujoco]"
```

On Windows, use the following command instead.

```bat
py -3.12 -m pip install --user numpy scipy mujoco roboticstoolbox-python ur_rtde robot_descriptions "xacrodoc[mujoco]"
```

For the remaining commands, replace `python` with `python3.12` on Linux and macOS, or `py -3.12` on Windows. On macOS, also make sure the user scripts directory reported by pip is on PATH so that `mjpython` can be found.

Continue to the [installation check](#check-the-installation).

</details>

### Check the installation

Run the environment check.

```text
python test_environment.py
```

The script prints the Python executable, version and import results for the main libraries. With venv, the executable path should point inside the `ur3` folder. This check does not connect to or move the robot. A missing module usually means a package was not installed or a different Python environment is active.

## Simulation examples

These examples run on the laptop without connecting to a physical robot. Use the same Python environment as the installation check and run one example at a time.

On macOS, examples that open the passive MuJoCo viewer need `mjpython`, which is installed with MuJoCo. The commands below account for this [viewer requirement](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).

### Check the model

[Example 03](examples/03_check_velocity_model.py) prints the model dimensions, actuator names and command limits. This checks that the simulation model can be loaded without opening a viewer.

```text
python examples/03_check_velocity_model.py
```

### Try a wrist movement

[Example 04](examples/04_joint_velocity_sim.py) opens the MuJoCo viewer and commands a small rotation of the last wrist joint. The terminal compares the expected and measured rotation.

On Linux and Windows

```text
python examples/04_joint_velocity_sim.py
```

On macOS

```bash
mjpython examples/04_joint_velocity_sim.py
```

### Explore the robot backend

[Example 05](examples/05_test_mujoco_backend.py) performs a wrist movement through the project's `MuJoCoUR3e` class. Read this example to see how to send a velocity command and read the joint positions through that interface.

On Linux and Windows

```text
python examples/05_test_mujoco_backend.py
```

On macOS

```bash
mjpython examples/05_test_mujoco_backend.py
```

Once the simulation works, continue to [Hardware setup](#hardware-setup).

## Hardware setup

### Dangerous things

Keep the workspace clear. Avoid collisions between robot links and check that the cable on top has enough slack throughout the movement.

### Borrow the device

At the Mechatronics Lab, scan the QR code and sign in to Power Apps. Register the device loan, then collect the key to borrow the device.

### Connect the laptop and start the controller

1. Connect the power cable and connect the network cable to the laptop. Route the cables underneath the device, clear of walkways and trolley wheels, to prevent trip hazards and cable damage.
2. Open the laptop's wired Ethernet settings and select manual IPv4 configuration. Set the IP address to `192.168.0.100`, subnet mask to `255.255.255.0` and gateway to `192.168.0.1`. If a prefix length is requested instead of a subnet mask, enter `24`. These values are the same on Linux, Windows and macOS. Apply them to Ethernet only. Leave the WiFi settings unchanged.
3. Turn on the switch on the power cable at the bottom of the platform. Get the touchscreen panel from the rear door and press its power button. Wait for the screen to finish starting up.
4. Find the robot IP printed underneath the large number next to the arm. Open a terminal and run the command below, replacing ROBOT_ADDRESS with that IP.

On Linux and macOS

```bash
ping -c 4 ROBOT_ADDRESS
```

On Windows

```bat
ping -n 4 ROBOT_ADDRESS
```

A reply confirms basic network connectivity. If there is no reply, check the cable, Ethernet settings and robot IP before continuing.

### Prepare the arm for control

1. On the panel, select Installation → Tool/IO and confirm that OnRobot is selected.
2. Press the initialisation button in the bottom left corner. Select ON, then START to release the joint brakes. Wait until the arm is fully started and the status is green, then select EXIT.
3. Use the Local or Remote selector at the top of the screen. In Local mode, use Move to position the arm manually, or Home to return to the configured home position.
4. Select Remote to control the robot from Python.

The panel's program status shows Stopped when the program is stopped. Running means the robot controller is executing a script.

## Robot examples

Complete the hardware setup and select Remote before running these examples.

### Read the robot state

[Example 09](examples/09_real_ur_read_only.py) reads the joint positions without commanding movement. Replace ROBOT_ADDRESS below with the robot IP from the label.

```text
python examples/09_real_ur_read_only.py --ip ROBOT_ADDRESS
```

Confirm that joint positions are displayed before continuing to a movement test. The laptop address `192.168.0.100` belongs only in the Ethernet settings.

### Prepare a movement test

Set `ROBOT_IP` in the selected script to the robot address. Check `TEST_SPEED` and `TEST_DURATION`, then run one test at a time. Movement starts automatically after connection.

### Rotate the last wrist joint

Run [Example 12](examples/12_real_speedj_minimal_test.py) to rotate the last wrist joint.

```text
python examples/12_real_speedj_minimal_test.py
```

### Move the elbow

[Example 13](examples/13_test_real_backend_velocity.py) is a separate elbow movement test through `qd_command[2]`. Its printed wrist label is incorrect.

```text
python examples/13_test_real_backend_velocity.py
```

### Control the RG2 gripper

[Example 15](examples/15_real_gripper_control.py) opens an RTDE control session so the external-control program can write RG2 commands. It keeps that session active while you control the gripper, but does not send robot-arm motion commands.

```text
python examples/15_real_gripper_control.py --ip ROBOT_ADDRESS
```

Press `Z` to close to 10 mm, `X` to open to 80 mm, and `Esc` to exit. Each command uses 20 N of gripping force.

### Make movement easier to see

If movement is hard to see, first confirm the robot IP and observe the commanded joint. A small increase in `TEST_DURATION` makes the movement larger at the same speed. Check clearance and cable slack before increasing the duration.

## Returning to the lab

Open a terminal in the project folder. With venv, activate the existing environment.

On Linux and macOS

```bash
source ur3/bin/activate
```

On Windows, use Command Prompt

```bat
ur3\Scripts\activate.bat
```

With the alternative installation, use the same Python installation without activating venv. The installed libraries remain available between sessions.

Continue with [Hardware setup](#hardware-setup), then follow the [robot examples](#robot-examples). Repeat the [installation check](#check-the-installation) if Python or the installed libraries have changed.

After finishing with the scripts, run `deactivate` to leave an active virtual environment. This only changes the terminal's Python environment.

## Working with RealSense

Use the `RealSense-Lab` branch for the Intel RealSense D435i. It contains a separate camera project with its own installation instructions. Switching branches changes the README and examples in the project folder.

Run these commands from the project folder in a copy cloned with Git. Commit or stash any local changes first.

```text
git fetch origin
git switch RealSense-Lab
```

The first command retrieves the latest branch information from GitHub. The second switches to the camera project. Open the [RealSense README](https://github.com/UTS-RI/UR3_RG2_rosbridge/tree/RealSense-Lab#readme) and follow its setup instructions before running the camera examples.

The branch contains 15 examples and reusable functions in `src/realsense_lab/`.

* Examples 01 and 04 show colour images and depth, including a live view.
* Examples 02 and 03 display point clouds coloured by depth or image colour.
* Examples 05, 12 and 13 save data, record camera streams and play recordings back.
* Example 14 reads existing camera calibration data and measures a 3D distance. It does not calibrate the camera relative to the robot.

Several examples also provide a demo mode with artificial data for practice without a camera. The camera README lists the available modes and additional examples.

To return to the robot project, run the following command.

```text
git switch main
```

## Camera calibration note

The original project notes suggest an ArUco marker on the end effector for camera calibration. The `main` branch does not include camera calibration code or a calibration procedure.

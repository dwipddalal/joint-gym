import pybullet as p
import pybullet_data
import numpy as np
import math
import time
import matplotlib.pyplot as plt


def forward_kinematics(theta1, theta2, l1, l2):
    x = l1*math.cos(theta1) + l2*math.cos(theta1 + theta2)
    y = l1*math.sin(theta1) + l2*math.sin(theta1 + theta2)
    return [x, y]


def inverse_kinematics(x, y, l1, l2, branch=1):
    a = 2*x*l2
    b = 2*y*l2
    c = l1*l1 - x*x - y*y - l2*l2
    psi = math.atan2(b, a)
    d = -c/math.sqrt(a*a + b*b)

    if (d < -1) or (d > 1):
        print("Position out of workspace.")
        return False, [0, 0]
    if branch == 1:
        theta12 = psi + math.acos(-c/math.sqrt(a*a + b*b))
    else:
        theta12 = psi - math.acos(-c/math.sqrt(a*a + b*b))

    theta1 = math.atan2((y - l2*math.sin(theta12))/l1,
                        (x - l2*math.cos(theta12))/l1)
    return True, [theta1, theta12-theta1]


# Use p.GUI to create a GUI to render the simulation
client = p.connect(p.GUI)  # or p.GUI

# Set the search path to find the plane.urdf file
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")

# Load the URDF of the robot
robot = p.loadURDF("planar_2R_robot.urdf")

# Set the Gravity vector, that is gravity in negative z direction
p.setGravity(0, 0, -9.81, physicsClientId=client)

# Set the simulation time-step
p.setTimeStep(0.001)  # The lower this is, more accurate the simulation

# This step is required to enable torque control.
p.setJointMotorControl2(robot, 1, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot, 2, p.VELOCITY_CONTROL, force=0)

# Kinematics for serial-2R
p1 = np.array([1.0, 0.5])
p2 = np.array([0.5, 1.0])
pt_des = p1  # or p2

valid, [theta1, theta2] = inverse_kinematics(pt_des[0], pt_des[1], 1, 1)


dt = 0.001  # simulation time-step
p_gain = 200  # Proportional gain
d_gain = 50  # Derivative gain
error = 0
error_old = 0
desired_pos = np.array([theta1, theta2])
for _ in range(1000):
    pos1, _, _, _ = p.getJointState(robot, 1)
    pos2, _, _, _ = p.getJointState(robot, 2)
    pos = np.array([pos1, pos2])
    error_old = error
    error = desired_pos - pos
    error_d = (error - error_old)/dt
    control_force = p_gain * error + d_gain * error_d
    p.setJointMotorControlArray(
        robot, [1, 2], p.TORQUE_CONTROL, forces=control_force)
    p.stepSimulation()
    time.sleep(1./240.)


# Check if the robot has reached the desired position
pos1, _, _, _ = p.getJointState(robot, 1)
pos2, _, _, _ = p.getJointState(robot, 2)
pt_act = forward_kinematics(pos1, pos2, 1, 1)

print("Desired position: ", pt_des)
print("Actual position: ", pt_act)

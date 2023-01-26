# The PyBullet physics simulation library
import pybullet as p
import pybullet_data

# Numpy for numerical calculations and manipulations
import numpy as np
import math
import time
import matplotlib.pyplot as plt

# Use p.GUI to create a GUI to render the simulation
client = p.connect(p.GUI)  # or p.GUI


# Load the URDF of the plane that forms the ground
# Set the search path to find the plane.urdf file
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")


# Load the URDF of the robot
robot = p.loadURDF("planar_2R_robot.urdf")

# Set the Gravity vector
# that is gravity in negative z direction
p.setGravity(0, 0, -9.81, physicsClientId=client)

# Set the simulation time-step
p.setTimeStep(0.001)  # The lower this is, more accurate the simulation

# This step is required to enable torque control.
p.setJointMotorControl2(robot, 1, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot, 2, p.VELOCITY_CONTROL, force=0)

# Create a Proportional control loop to regulate the position of a single joint

# Define a sinusoidal trajectory
dt = 0.001  # Simulation time-step
f = 1.0  # Frequency of oscillation (1 Hz)
omega = 2*math.pi*f  # Angular frequency
theta0 = 0  # Start position
p_des = np.zeros(10000)
for i in range(10000):
    t = i*dt
    p_des[i] = np.sin(theta0 + omega*t)

# Define the gains
p_gain = 1000  # Proportional gain
d_gain = 500  # Derivative gain

error = 0
error_old = 0

pos1 = []
cf = []

# Run the control loop
for i in range(10000):

    # Get the joint state
    p_act, _, _, _ = p.getJointState(robot, 1)

    # Calculate the control input
    error_old = error
    error = p_des[i] - p_act
    error_d = (error - error_old)/dt
    control_force = p_gain * error + d_gain * error_d  # PD control
    # Saturation; to model the torque limit of the motors
    control_force = np.clip(control_force, -50, 50)

    # Run the simulation for one time-step
    p.setJointMotorControl2(robot, 1, p.TORQUE_CONTROL, force=control_force)
    p.stepSimulation()
    time.sleep(1./240.)

    # Store the data for plotting
    pos1.append(p_act)
    cf.append(control_force)

# Plot the results
plt.figure(2)
plt.plot(pos1, label="Actual position")
plt.plot(p_des, label="Desired position")
plt.ylim([-2, 2])
# plt.plot(cf, label="Control Input")
plt.legend()
plt.show()

# Disconnect from the physics server
p.disconnect()

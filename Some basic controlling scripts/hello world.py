# The PyBullet physics simulation library
import pybullet as p
import pybullet_data
import time

# Numpy for numerical calculations and manipulations
import numpy as np
import math

# Matplotlib to create the necessary plots
import matplotlib.pyplot as plt

# Use p.DIRECT to connect to the server without rendering a GUI
# Use p.GUI to create a GUI to render the simulation
client = p.connect(p.GUI) # or p.GUI


# Load the URDF of the plane that forms the ground
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Set the search path to find the plane.urdf file
# plane = p.loadURDF("plane.urdf")


# Load the URDF of the robot
robot = p.loadURDF("spatial_3R_robot.urdf")

# Get the number of joints in the robot
num_joints = p.getNumJoints(robot)
print("--------------------------")
print(num_joints)
print("---------------------------------")
# Set the initial joint positions

def change(arr):
    x = arr[0]
    y = arr[1]
    z = arr[2]
    return [x+0.01, y+0.01, z+0.01]

joint_positions = [0,0,0]

def go(joint_positions):
    joint_positions = change(joint_positions)
    for i in range(num_joints):
        p.resetJointState(robot, i, joint_positions[i])  
    return joint_positions

# Run the simulation for 1000 timesteps
for i in range(5000):
    joint_positions = go(joint_positions)
    p.stepSimulation()
    time.sleep(1./240.)

# Clean up
p.disconnect()
This code is a simulation of a 2-link robotic arm using the PyBullet physics library. It uses Proportional-Derivative (PD) control to regulate the position of a single joint of the robot. The simulation is displayed in a GUI window.

The code first imports the necessary libraries (PyBullet, numpy, and matplotlib). It then connects to the PyBullet GUI and loads the URDF (Unified Robot Description Format) of a plane and a 2-link robotic arm into the simulation. The gravity vector is set in the negative z direction and the simulation time-step is set to 0.001 seconds. The code then enables torque control for the two joints of the robotic arm.

Next, the code defines a sinusoidal trajectory for the position of the first joint and creates a PD control loop to regulate the position of that joint. The loop runs for 10000 iterations, where in each iteration, it gets the actual position of the joint, calculates the control input using PD control with the defined proportional gain (1000) and derivative gain (500), saturates the control input to model the torque limit of the motors, and runs the simulation for one time-step. The actual position and control input are stored in arrays for plotting at the end.

Finally, the code plots the actual position, desired position, and control input of the first joint, and the simulation is disconnected.

<img width="431" alt="image" src="https://user-images.githubusercontent.com/91228207/214959828-cebd8976-53b7-4ec6-a10e-e9702d159540.png">

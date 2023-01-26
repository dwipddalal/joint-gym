## Planar 2R Robot Kinematics

This code contains two functions forward_kinematics and inverse_kinematics, which are used to calculate the forward and inverse kinematics of a 2-link planar robot, respectively.


`forward_kinematics(theta1, theta2, l1, l2)`

This function takes four inputs:

- `theta1`: the angle of the first link with respect to the x-axis.
- `theta2`: the angle of the second link with respect to the first link.
- `l1`: the length of the first link.
- `l2`: the length of the second link.

It returns a list containing the x and y coordinates of the end-effector of the robot.

`inverse_kinematics(x, y, l1, l2, branch=1)`

This function takes four inputs:

- `x`: the x-coordinate of the end-effector of the robot.
- `y`: the y-coordinate of the end-effector of the robot.
- `l1`: the length of the first link.
- `l2`: the length of the second link.
- `branch`: an optional parameter (default value is 1) that selects the branch of the inverse kinematics solution to use.

It returns a tuple containing a boolean value indicating if the position is within the robot's workspace and a list containing the angles of the first and second links.

The code then uses PyBullet to simulate the robot's movement and control its movement using a proportional-derivative control loop with a specified time step. The robot's position and the desired position are continuously compared and the error is used to calculate the control force applied to the joints. The simulation runs for 1000 iterations, after which the actual position of the end-effector is compared to the desired position and printed.

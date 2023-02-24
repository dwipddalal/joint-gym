## Reward function used here:
give a reward of +1 for each time step that the pole remains upright, and a reward of 0 when the pole falls over or the cart goes out of bounds. This reward scheme incentivizes the learning algorithm to keep the pole balanced for as long as possible.

## Discounted reward (our target variable)
The discounted reward at time step t, denoted by G(t), can be calculated using the following formula:

G(t) = r(t) + γ * r(t+1) + γ^2 * r(t+2) + ... + γ^(T-t-1) * r(T-1) + γ^(T-t) * r(T)

where r(t) is the reward at time step t, γ is the discount factor, T is the maximum time step in the episode, and t ranges from 0 to T-1.

Here, the discounted reward is calculated in the same way as described above, using the rewards provided by the reward function at each time step. The discounted reward is used as the target variable for the network in order to learn the optimal policy for balancing the pole. The network is trained MSE loss function (regresion loss) that minimizes the difference between the predicted discounted reward and the actual discounted reward obtained during the episode.



if the agent selects an action that moves the cart to the right and this action leads to the pole remaining upright for another time step, the agent would receive a positive reward of +1. If, on the other hand, the agent selects an action that causes the pole to fall over, the agent would receive a negative reward of 0

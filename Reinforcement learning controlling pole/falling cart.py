import gym
import time 

env = gym.make('CartPole-v1')

observation = env.reset()

done = False

while not done:
    
    env.render()
    action = env.action_space.sample() # sample a random action from the action space
    observation, reward, done, info, _ = env.step(action)
    time.sleep(1)
    print(env.step(action))
    
    
env.close()

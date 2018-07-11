import gym
import numpy as np

env = gym.make('GuessingGame-v0')
#env = gym.make('MountainCar-v0')
env.reset()
print(env.action_space)
print(env.observation_space)

done = False
while not done:
    gu = input("make a guess:")
    observation, reward, done, info = env.step(np.asarray([float(gu)])) # take a random action
    print("result: " + str(observation))
    print("reward: " + str(reward))
env.reset()
print(done)

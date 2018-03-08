import gym
import numpy as np
import time
import tensorflow as tf
import random
from collections import deque
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

env = gym.make('Pong-v0')
env.reset()

done = False
#env.render()
t = 0
while not done:
    act = env.action_space.sample()
    newObs, r, done, _ = env.step(act)
    t += 1
    st = newObs[35:195, :, 2]
    st[np.where(st<=17)] = 0
    st[np.where(st>17)] = 255
    cv2.imshow("tmp", st)
    cv2.waitKey()

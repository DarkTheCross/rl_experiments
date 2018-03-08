import gym
import numpy as np
import time
import tensorflow as tf
import random
from collections import deque
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class cpsolver():

    def __init__( self, gamma = 0.95, epsilon = 1.0, eps_decay = 0.99, eps_min = 0.0, lr = 0.001, lr_decay = 0.98):
        self.x = tf.placeholder(tf.float32, shape=[None, 160, 160, 3])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 6])
        self.memory = deque(maxlen=100000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.lr = lr
        #self.lr_decay = lr_decay
        self.conv1 = tf.layers.conv2d(self.x, 4, 5, (2,2), activation=tf.nn.relu, padding='valid')
        self.mp1 = tf.layers.max_pooling2d(self.conv1, strides=(2,2), pool_size=(3,3), padding='valid')
        self.conv2 = tf.layers.conv2d(self.mp1, 4, 5, (2,2), activation=tf.nn.relu, padding='valid')
        self.mp2 = tf.layers.max_pooling2d(self.conv2, strides=(2,2), pool_size=(3,3), padding='valid')
        self.conv3 = tf.layers.conv2d(self.mp2, 4, 3, (2,2), activation=tf.nn.relu, padding='valid')
        self.flt = tf.contrib.layers.flatten(self.conv3)
        self.nnResult = tf.contrib.layers.fully_connected(self.flt, 6)
        self.err = tf.losses.mean_squared_error(self.nnResult, self.y_)
        self.train_step =  tf.train.AdamOptimizer(self.lr).minimize(self.err)
        self.env = gym.make('Pong-v0')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def decision(self, eps, state):
        if random.random() < eps:
            return self.env.action_space.sample()
        else:
            tmpres1 = self.nnResult.eval(feed_dict={self.x: [state]})
            return np.argmax(tmpres1)
            #print(tmpres)

    def learn(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        x_batch = np.zeros((len(minibatch), 160, 160, 3), dtype=np.float32)
        y_batch = np.zeros((len(minibatch), 6), dtype=np.float32)
        iter_idx = 0
        for state, action, reward, next_state, done in minibatch:
            x_batch[iter_idx, :, :, :] = state[:, :, :]
            tmpRes = self.nnResult.eval(feed_dict={self.x: [next_state]})
            #print(tmpRes)
            if done:
                tmpRes[0][action] = reward
            else:
                tmpRes[0][action] = reward + self.gamma * np.max(tmpRes)
            y_batch[iter_idx] = tmpRes
            iter_idx += 1
        self.train_step.run(feed_dict = {self.x: x_batch, self.y_: y_batch})

    def run(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                mem_rec = []
                obs = self.env.reset()
                st = np.zeros((160, 160, 3), dtype=np.uint8)
                st[:, :, 2] = obs[35:195, :, 2]
                st[np.where(st<=17)] = 0
                st[np.where(st>17)] = 255
                done = False
                tick_num = 0
                while not done:
                    tick_num += 1
                    act = self.decision(self.epsilon, st)
                    newObs, r, done, _ = self.env.step(act)
                    #self.env.render()
                    newSt = np.copy(st)
                    newSt[:, :, 0] = newSt[:, :, 1]
                    newSt[:, :, 1] = newSt[:, :, 2]
                    newSt[:, :, 2] = newObs[35:195, :, 2]
                    newSt[np.where(newSt<=17)] = 0
                    newSt[np.where(newSt>17)] = 255
                    self.remember(st, act, r, newSt, done)
                    st = newSt
                mem_rec.append(tick_num)
                print(tick_num)
                if self.epsilon > self.eps_min:
                    self.epsilon = self.epsilon * self.eps_decay
                self.learn(50)
                if i % 10 == 0:
                    print("Mean ticks for last 10 trials: " + str( np.mean(mem_rec) ))
                    #print("Mean ticks for last 100 trials: " + str( np.mean(qu) ))
                    #print(str( len(qu) ))
                    #print("eps: " + str(self.epsilon))
                    mem_rec = []

if __name__ == "__main__":
    cps = cpsolver()
    cps.run()

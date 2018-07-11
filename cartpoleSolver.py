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
        self.x = tf.placeholder(tf.float32, shape=[None, 5])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])
        self.memory = deque(maxlen=100000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.lr = lr
        #self.lr_decay = lr_decay
        self.Q_Weights = {}
        self.Q_Weights["W1"] = tf.Variable(tf.random_normal([5, 64], stddev=0.35),name="Q_W1")
        self.Q_Weights["b1"] = tf.Variable(tf.random_normal([64], stddev=0.35),name="Q_b1")
        self.Q_Weights["W2"] = tf.Variable(tf.random_normal([64, 32], stddev=0.35),name="Q_W2")
        self.Q_Weights["b2"] = tf.Variable(tf.random_normal([32], stddev=0.35),name="Q_b2")
        self.Q_Weights["W3"] = tf.Variable(tf.random_normal([32, 1], stddev=0.35),name="Q_W3")
        self.Q_Weights["b3"] = tf.Variable(tf.random_normal([1], stddev=0.35),name="Q_b3")
        self.x1 = tf.nn.tanh(tf.matmul(self.x, self.Q_Weights["W1"]) + self.Q_Weights["b1"])
        self.x2 = tf.nn.relu(tf.matmul(self.x1, self.Q_Weights["W2"]) + self.Q_Weights["b2"])
        self.nnResult = tf.matmul(self.x2, self.Q_Weights["W3"]) + self.Q_Weights["b3"]
        self.err = tf.losses.mean_squared_error(self.nnResult, self.y_)
        self.train_step =  tf.train.AdamOptimizer(self.lr).minimize(self.err)
        self.env = gym.make('CartPole-v0')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def decision(self, eps, state):
        if random.random() < eps:
            return self.env.action_space.sample()
        else:
            state0 = np.zeros((5,), dtype=np.float32)
            state0[:4] = state
            tmpres0 = self.nnResult.eval(feed_dict={self.x: [state0]})
            state0[4] = 1
            tmpres1 = self.nnResult.eval(feed_dict={self.x: [state0]})
            if tmpres0 > tmpres1:
                return 0
            else:
                return 1

    def learn(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        x_batch = np.zeros((len(minibatch), 5), dtype=np.float32)
        y_batch = np.zeros((len(minibatch), 1), dtype=np.float32)
        iter_idx = 0
        for state, action, reward, next_state, done in minibatch:
            state0 = np.zeros((5,), dtype=np.float32)
            state0[:4] = state
            state0[4] = action
            x_batch[iter_idx, :] = state0
            tmpRes = 0
            #print(tmpRes)
            if done:
                tmpRes = reward
            else:
                state1 = np.zeros((5,), dtype=np.float32)
                state1[:4] = next_state
                newRes1 = self.nnResult.eval(feed_dict={self.x: [state1]})
                state1[4] = 1
                newRes2 = self.nnResult.eval(feed_dict={self.x: [state1]})
                tmpRes = reward + self.gamma * max(newRes1, newRes2)
            y_batch[iter_idx] = tmpRes
            iter_idx += 1
        #print(x_batch)
        #print(y_batch)
        #self.train_step =  tf.train.GradientDescentOptimizer(self.lr).minimize(self.err)
        self.train_step.run(feed_dict = {self.x: x_batch, self.y_: y_batch})
        #self.lr *= self.lr_decay
        #print(self.Q_Weights["b3"].eval())

    def run(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            qu = []
            for i in range(1000):
                mem_rec = []
                obs = self.env.reset()
                done = False
                tick_num = 0
                while not done:
                    tick_num += 1
                    act = self.decision(self.epsilon, obs)
                    newObs, r, done, _ = self.env.step(act)
                    self.remember(obs, act, r, newObs, done)
                    obs = newObs
                mem_rec.append(tick_num)
                if len(qu)<100:
                    qu.append(tick_num)
                else:
                    del qu[0]
                    qu.append(tick_num)
                    #print(qu)
                if np.mean(qu) > 195:
                    print("Solved after " + str(i) + " iterations. ")
                    break
                if self.epsilon > self.eps_min:
                    self.epsilon = self.epsilon * self.eps_decay
                self.learn(100)
                if i % 50 == 0:
                    print("Mean ticks for last 50 trials: " + str( np.mean(mem_rec) ))
                    #print("Mean ticks for last 100 trials: " + str( np.mean(qu) ))
                    #print(str( len(qu) ))
                    #print("eps: " + str(self.epsilon))
                    mem_rec = []

if __name__ == "__main__":
    cps = cpsolver()
    cps.run()

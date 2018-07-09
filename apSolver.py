import numpy as np
import tensorflow as tf
from ap import ApplePicker
import random
from collections import deque
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class apSolver:
    def __init__(self, eps_decay = 0.996, gamma = 0.9, eps_min = 0.01):
        self.env = ApplePicker()
        self.eps = 1
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.x = tf.placeholder(tf.float32, shape=[None, 7, 7])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 3])
        self.x_flatten = tf.contrib.layers.flatten(self.x)
        self.x_1 = tf.contrib.layers.fully_connected(self.x_flatten, 64)
        self.x_2 = tf.contrib.layers.fully_connected(self.x_1, 128)
        self.x_3 = tf.contrib.layers.fully_connected(self.x_2, 64)
        self.res_cur = tf.contrib.layers.fully_connected(self.x_3, 3, activation_fn = None)
        self.err = tf.losses.huber_loss(self.res_cur, self.y_)
        self.train_step =  tf.train.AdamOptimizer(1e-2).minimize(self.err)
        self.memory = deque(maxlen=100000)

    def decision(self, eps, state):
        if random.random() > eps:
            res_c = self.res_cur.eval(feed_dict = {self.x: [state]})
            #print(np.argmax(res_c))
            return np.argmax(res_c)
        else:
            act = random.randint(0, 2)
            return act

    def train(self, batch_size):
        bs = min(len(self.memory), batch_size)
        tmpx = np.zeros((bs, 7, 7), dtype=np.float32)
        tmpy = np.zeros((bs, 3), dtype=np.float32)
        minibatch = random.sample(self.memory, bs)
        iter_idx = 0
        for state, act, rwd, next_state, done in minibatch:
            y_res = self.res_cur.eval(feed_dict = {self.x: [state]})
            if done:
                y_res[0][act] = rwd
            else:
                y_res1 = self.res_cur.eval(feed_dict = {self.x: [next_state]})
                y_res[0][act] = rwd + self.gamma * np.max(y_res1)
            tmpx[iter_idx, :, :] = state
            tmpy[iter_idx, :] = y_res[0]
            iter_idx += 1
        self.train_step.run(feed_dict={self.x: tmpx, self.y_: tmpy})

    def run(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            obs = self.env.reset()
            done = False
            memrec = []
            for i in range(1500):
                count = 0
                imgid = 0
                while not done:
                    act = self.decision(self.eps, obs)
                    newObs, rwd, done = self.env.step(act)
                    self.memory.append((obs, act, rwd, newObs, done))
                    if i % 30 == 0 and not i == 0:
                        obsImg = cv2.resize(obs, (420, 420), interpolation = cv2.INTER_LINEAR)*255
                        cv2.imwrite("img/"+str(i)+"_" + str(imgid) + ".png", obsImg)
                        imgid += 1
                    count += rwd
                    obs = newObs
                if i % 30 == 0 and not i == 0:
                    obsImg = cv2.resize(obs, (420, 420))
                    cv2.imwrite("img/"+str(i)+"_" + str(imgid) + ".png", obsImg)
                memrec.append(count)
                done = False
                obs = self.env.reset()
                if self.eps > self.eps_min:
                    self.eps *= self.eps_decay
                else:
                    self.eps = 0
                self.train(100)
                if i % 30 == 0 and not i == 0:
                    print("avg at " + str(i) + ", eps = " + str(self.eps) + " : " + str( np.average(memrec) ))
                    memrec = []

if __name__ == "__main__":
    aps = apSolver()
    aps.run()

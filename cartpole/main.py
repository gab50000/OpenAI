#!/usr/bin/env python3

import sys
import pickle
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

import gym
import numpy as np
import tensorflow as tf


logging.getLogger("gym").setLevel(logging.WARN)


def NeuralNet(input_size: int, hidden_size: int, output_size: int):
    input_ = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name="input")
    output = tf.placeholder(dtype=tf.float32, shape=[None, output_size], name="output")
    W1 = tf.Variable(tf.random_normal([input_size, hidden_size]))
    b1 = tf.Variable(tf.random_normal([hidden_size]))
    W2 = tf.Variable(tf.random_normal([hidden_size, output_size]))
    b2 = tf.Variable(tf.random_normal([output_size]))
    hidden = tf.tanh(tf.matmul(input_, W1) + b1, "hidden")
    predicted_q = tf.matmul(hidden, W2) + b2
    action = tf.argmax(predicted_q, axis=1)

    return input_, output, predicted_q, action


# R_t = r_t + gamma * R_{t+1}
# R_t - gamma * R_{t+1} - r_t = 0
def determine_cumulative_reward(rewards, *, gamma):
    R_t = np.zeros(len(rewards))
    R_t[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        R_t[i] = rewards[i] + gamma * R_t[i + 1]
    return R_t


def main():
    input_size, hidden_size, output_size = 4, 10, 2
    gamma = 0.9

    max_steps = 1000
    epochs = 1000
    
    input_, output, predicted_q, predicted_action = NeuralNet(input_size, hidden_size, output_size)
    rewards_ = tf.placeholder(dtype=tf.float32, shape=4)
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    #loss = output[:-1] - gamma * output[1:] - rewards_
    # train = optimizer.minimize(loss)

    env = gym.make('CartPole-v0')

    
    observables_total = []
    rewards_total = []

    for epoch in range(epochs):
        observables = []
        actions = []
        rewards = []
        net_outputs = []
        logger.info("Epoch: {}".format(epoch))
        env.reset()
        obs = env.observation_space.sample()
        for step in range(1000):
            env.render()
            guessed_action, = session.run(predicted_action, {input_: obs[None, :]})
            obs, reward, done, info = env.step(guessed_action)
            actions.append(guessed_action)
            observables.append(obs)
            logger.info("Step".format(step))
            if done:
                reward = -1
            rewards.append(reward)
            net_outputs.append(guessed_action)
            if done:
                break

        out_arr = np.array(net_outputs)
        cumulative_rewards = determine_cumulative_reward(rewards, gamma=gamma)
        rewards = np.array(rewards)
        #guessed_actions = out_arr[range(len(out_arr)), actions]
        delta = cumulative_rewards[:-1] - gamma * cumulative_rewards[1:] - rewards[:-1]
        logger.debug("Observables: {}".format(observables))
        logger.debug("Actions: {}".format(actions))
        logger.debug("Rewards: {}".format(cumulative_rewards))
        #  print("Guessed:", guessed_actions)
        logger.debug("Diffs: {}".format(delta))

        rewards_total.append(delta)
        
        with open("data", "wb") as f:
            pickle.dump({"observables": observables_total, "rewards": rewards_total}, f)


if __name__ == "__main__":
    main()

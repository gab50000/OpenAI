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
    input_nodes = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name="input")
    output_nodes = tf.placeholder(dtype=tf.float32, shape=[None, output_size], name="output")
    W1 = tf.Variable(tf.random_normal([input_size, hidden_size]))
    b1 = tf.Variable(tf.random_normal([hidden_size]))
    W2 = tf.Variable(tf.random_normal([hidden_size, output_size]))
    b2 = tf.Variable(tf.random_normal([output_size]))
    hidden = tf.tanh(tf.matmul(input_nodes, W1) + b1, "hidden")
    net = tf.nn.softmax(tf.matmul(hidden, W2) + b2)
    return input_nodes, output_nodes, net


# R_t = r_t + gamma * R_{t+1}
# R_t - gamma * R_{t+1} - r_t = 0
def determine_future_reward(rewards, *, gamma):
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
    
    input_nodes, output_nodes, net = NeuralNet(input_size, hidden_size, output_size)
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

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
        for step in range(1000):
            env.render()
            action = env.action_space.sample()  # take a random action
            obs, reward, done, info = env.step(action)
            actions.append(action)
            observables.append(obs)
            logger.info("Step".format(step))
            if done:
                reward = -1
            rewards.append(reward)
            guess = session.run(net, {input_nodes: obs[None, :]})
            net_outputs.append(guess.squeeze())
            if done:
                break

        out_arr = np.array(net_outputs)
        future_rewards = determine_future_reward(rewards, gamma=gamma)
        rewards = np.array(rewards)
        guessed_rewards = out_arr[range(len(out_arr)), actions]
        delta = guessed_rewards[:-1] - gamma * guessed_rewards[1:] - rewards[:-1]
        logger.debug("Observables: {}".format(observables))
        logger.debug("Actions: {}".format(actions))
        logger.debug("Rewards: {}".format(future_rewards))
        #  print("Guessed:", guessed_rewards)
        logger.debug("Diffs: {}".format(delta))

        rewards_total.append(delta)
        
        with open("data", "wb") as f:
            pickle.dump({"observables": observables_total, "rewards": rewards_total}, f)


if __name__ == "__main__":
    main()

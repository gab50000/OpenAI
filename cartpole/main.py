#!/usr/bin/env python3

from functools import reduce
from operator import add
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import gym
import numpy as np
import tensorflow as tf


logging.getLogger("gym").setLevel(logging.WARN)


RENDER = True


def create_neural_net(input_size: int, hidden_size: int, output_size: int):
    input_ = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name="input")
    actual_reward = tf.placeholder(dtype=tf.float32, shape=[None, 1],
                                   name="actual_reward")
    W1 = tf.Variable(tf.random_normal([input_size, hidden_size]))
    b1 = tf.Variable(tf.random_normal([hidden_size]))
    W2 = tf.Variable(tf.random_normal([hidden_size, output_size]))
    b2 = tf.Variable(tf.random_normal([output_size]))
    hidden = tf.tanh(tf.matmul(input_, W1) + b1, "hidden")
    all_q = tf.matmul(hidden, W2) + b2
    predicted_actions = tf.argmax(all_q, axis=1)
    predicted_q = tf.reduce_max(all_q, axis=1)
    regularizer = reduce(add, (tf.nn.l2_loss(w) for w in (W1, W2, b1, b2)))
    weights = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

    return input_, actual_reward, predicted_q, predicted_actions, regularizer, weights


# R_t = r_t + gamma * R_{t+1}
# R_t - gamma * R_{t+1} - r_t = 0
def determine_cumulative_reward(rewards, *, gamma):
    R_t = np.zeros(len(rewards))
    R_t[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        R_t[i] = rewards[i] + gamma * R_t[i + 1]
    return R_t


def main():
    logfile = open("log.out", "w")
    input_size, hidden_size, output_size = 4, 50, 2
    gamma = 0.9
    beta = 0.01

    max_steps = 1000
    epochs = 1000
    steps = 10000

    input_, actual_reward, predicted_q, predicted_action, regularizer, weights = \
        create_neural_net(input_size, hidden_size, output_size)
    session = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    init.run()

    optimizer = tf.train.MomentumOptimizer(0.02, momentum=0.5)
    #loss = tf.reduce_mean(tf.square(predicted_q - gamma * predicted_q - actual_reward))
    delta = predicted_q - gamma * predicted_q - actual_reward
    loss = tf.losses.huber_loss(delta, tf.zeros(shape=tf.shape(delta)))
    loss = tf.reduce_mean(loss + beta * regularizer)
    #loss = tf.reduce_mean(tf.square(predicted_q - actual_reward))
    train = optimizer.minimize(loss)
    session.run(tf.global_variables_initializer())

    env = gym.make('CartPole-v0')

    observables_total = []
    rewards_total = []

    for epoch in range(epochs):
        observables = []
        guessed_actions = []
        rewards = []
        guessed_cumulative_rewards = []

        net_outputs = []
        logger.info("Epoch: {}".format(epoch))
        env.reset()
        obs = env.observation_space.sample()
        observables.append(obs)
        for step in range(steps):
            if RENDER:
                env.render()
            gcr, guessed_action = session.run([predicted_q, predicted_action],
                                              {input_: obs[None, :]})
            guessed_action = guessed_action[0]
            guessed_cumulative_rewards.append(gcr.squeeze())
            obs, reward, done, info = env.step(guessed_action)
            print(f"Obs = {obs}")
            guessed_actions.append(guessed_action)
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
        guessed_cumulative_rewards = np.array(guessed_cumulative_rewards)
        observables = np.array(observables)
        #guessed_actions = out_arr[range(len(out_arr)), guessed_actions]
        #rewards_of_chosen_actions = guessed_cumulative_rewards[range(guessed_cumulative_rewards.shape[0]), guessed_actions]
        #delta = rewards_of_chosen_actions[:-1] - gamma * rewards_of_chosen_actions[1:] - rewards[:-1]
        session.run(train, {input_: observables, actual_reward: rewards[:, None]})
        losstmp = session.run(loss, {input_: observables, actual_reward:rewards[:, None]})
        W1 = session.run(weights["W1"])
        W2 = session.run(weights["W2"])
        b1 = session.run(weights["b1"])
        b2 = session.run(weights["b2"])
        logger.info(f"Loss is {losstmp}")
        if any(np.isnan(x).any() for x in (W1, W2, b1, b2)):
            import ipdb; ipdb.set_trace()
        #print(f"Loss is {losstmp}", file=logfile, flush=True)
        print(losstmp, file=logfile, flush=True)
        logger.debug(f"Observables: {observables}")
        logger.debug(f"Actions: {guessed_actions}")
        logger.debug(f"Rewards: {cumulative_rewards}")
        #  print("Guessed:", guessed_actions)
        #logger.debug("Delta: {}".format(delta))

        #rewards_total.append(delta)

        #with open("data", "wb") as f:
        #    pickle.dump({"observables": observables_total, "rewards": rewards_total}, f)


if __name__ == "__main__":
    main()

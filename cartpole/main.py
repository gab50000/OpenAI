#!/usr/bin/env python3

import sys
import pickle
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

import gym
import numpy as np

sys.path.append("/home/kabbe/Code/Python/NeuroSimple/")

from neuro_simple.main import FFNQuadraticSigmoid

logging.getLogger("gym").setLevel(logging.WARN)
# R_t = r_t + gamma * R_{t+1}
# R_t - gamma * R_{t+1} - r_t = 0

def determine_future_reward(rewards):
    gamma = 0.9
    R_t = np.zeros(len(rewards))
    R_t[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        R_t[i] = rewards[i] + gamma * R_t[i + 1]
    return R_t


def main():
    input_size, hidden_size, output_size = 4, 10, 2
    
    max_steps = 1000
    epochs = 1000
    
    net = FFNQuadraticSigmoid((input_size, hidden_size, output_size))
    
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
            guess = net.propagate(obs)
            net_outputs.append(guess)
            if done:
                break

        out_arr = np.array(net_outputs)
        future_rewards = determine_future_reward(rewards)
        rewards = np.array(rewards)
        guessed_rewards = out_arr[range(len(out_arr)), actions]
        delta = guessed_rewards[:-1] - 0.9 * guessed_rewards[1:] - rewards[:-1]
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

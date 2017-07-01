#!/usr/bin/env python3

import sys

import gym
import torch
from torch.autograd import Variable
import numpy as np

sys.path.append("/home/kabbe/Code/Python/NeuroSimple/")

from neuro_simple.main import FeedForwardNetwork



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
    
    net = FeedForwardNetwork((input_size, hidden_size, output_size))
    
    env = gym.make('CartPole-v0')
    env.reset()

    actions = []
    rewards = []
    net_outputs = []

    for step in range(1000):
        env.render()
        action = env.action_space.sample()  # take a random action
        actions.append(action)
        obs, reward, done, info = env.step(action)
        print("Step", step, ":", obs, reward, done)
        if done:
            reward = -1
        rewards.append(reward)
        guess = net.propagate(obs)
        net_outputs.append(guess)
        import ipdb; ipdb.set_trace()
        if done:
            break

    net_outputs = np.array(net_outputs)
    future_rewards = determine_future_reward(rewards)
    guessed_rewards = net_outputs[range(len(net_outputs)), actions]
    print("Actions:", actions)
    print("Rewards:", future_rewards)
    print("Guessed:", guessed_rewards)
    print("Diffs:", guessed_rewards[:-1] - 0.9 * guessed_rewards[1:])


if __name__ == "__main__":
    main()

from itertools import count
import random

import gym
import numpy as np
import tensorflow as tf
import daiquiri


daiquiri.setup(level=daiquiri.logging.DEBUG)
logger = daiquiri.getLogger(__name__)


RENDER = False


class Neuro:
    def __init__(self):

        n_in = 4
        n_hidden = 3
        n_out = 2

        initializer = tf.contrib.layers.variance_scaling_initializer()
        self.input_ = tf.placeholder(dtype=tf.float32, shape=[None, n_in])
        hidden = tf.layers.dense(self.input_, n_hidden, activation=tf.nn.sigmoid)
        self.probs = tf.nn.softmax(tf.layers.dense(hidden, 2))
        self.gradients = tf.gradients(self.probs[0, 0], tf.trainable_variables())

    def run(self, input_arr):
        return self.sess.run(self.probs, feed_dict={self.input_: input_arr})

    @property
    def sess(self):
        if not hasattr(self, "_sess"):
            self._sess = tf.get_default_session()
            return self._sess

    def update_gradients(self, states, actions, scores):
        grads = self.sess.run(self.gradients, feed_dict={self.input_: states})
        import ipdb; ipdb.set_trace()


def calculate_mean_gradient(states, actions, input_, dp0dW):
    grads = [g.eval(feed_dict={input_: np.vstack(states[:-1])}) for g, v in dp0dW]
    import ipdb; ipdb.set_trace()


def evaluate_game(states, actions, immediate_rewards):
    evaluation = []
    dr = discounted_reward(immediate_rewards)
    return [(state, action, r) for state, action, r in zip(states, actions, dr)]


def discounted_reward(immediate_rewards, discount_factor=0.9):
    discounted_arr = np.zeros(len(immediate_rewards))
    discounted_arr[-1] = immediate_rewards[-1]
    for i in reversed(range(discounted_arr.size - 1)):
        discounted_arr[i] = immediate_rewards[i] + discount_factor * discounted_arr[i+1]
    return discounted_arr


def main():
    reward_gameover = -10
    reward_stillplaying = 1
    input_, neuro = Neuro()
    init = tf.global_variables_initializer()    

    env = gym.make('CartPole-v0')

    dp0dW = tf.gradients(neuro[0, 0], tf.trainable_variables())
    #  dp1dW = optimizer.compute_gradients(neuro[0, 1])
    updated_grads = [tf.placeholder(tf.float32, shape=weight.get_shape()) 
                     for weight in tf.trainable_variables()]
    training_op = [tf.assign(weight, update) 
                   for weight, update in zip(tf.trainable_variables(), updated_grads)]

    sess = tf.InteractiveSession()
    init.run()
    evaluations = []
    for training_episode in range(10000):
        obs = env.reset()
        states, actions, immediate_rewards = [obs], [], []

        for training_step in range(1000):
            env.render()
            pos, vel, angle, angle_vel = obs
            #  step = 1 if angle_vel >= 0 else 0
            step = random.randrange(0, 2)
            actions.append(step)
            obs, reward, done, info = env.step(step)
            states.append(obs)
            if done:
                immediate_rewards.append(reward_gameover)
                evalu = evaluate_game(sess, states, actions, immediate_rewards)
                evaluations += evalu
                break
            immediate_rewards.append(reward_stillplaying)


if __name__ == "__main__":
    main()

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
        scores = np.array(scores)
        scores = (scores - scores.mean()) / scores.std()
        training_step = 0.01
        sess = self.sess
        logger.debug("Calculating gradients")
        grads = [sess.run(self.gradients, feed_dict={self.input_: state[None, :]})
                 for state in states]
        
        weights = tf.trainable_variables()
        logger.debug("Calculating weight updates")
        for grad, action, score in zip(grads, actions, scores):
            sign = 1 - (2 * action)
            weights = [w + sign * np.sign(score) * training_step * g 
                       for w, g in zip(weights, grad)]
        logger.debug("Apply weight updates")
        sess.run([tf.assign(w, update) for w, update in zip(tf.trainable_variables(), weights)])
        logger.debug("New weights: %s", sess.run(tf.trainable_variables()))


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
    no_of_training_episodes = 10_000
    no_of_training_steps = 1000
    update_frequency = 100
    reward_gameover = -10
    reward_stillplaying = 1
    neuro = Neuro()
    init = tf.global_variables_initializer()    

    env = gym.make('CartPole-v0')

    sess = tf.InteractiveSession()
    init.run()
    evaluations = []
    for training_episode in range(1, no_of_training_episodes + 1):
        logger.info("Training episode %i", training_episode)
        obs = env.reset()
        states, actions, immediate_rewards = [obs], [], []

        for training_step in range(no_of_training_steps):
            if RENDER:
                env.render()
            #  pos, vel, angle, angle_vel = obs
            #  step = 1 if angle_vel >= 0 else 0
            step = np.argmax(neuro.run(obs[None, :]))
            actions.append(step)
            obs, reward, done, info = env.step(step)
            states.append(obs)
            if done:
                immediate_rewards.append(reward_gameover)
                evalu = evaluate_game(states, actions, immediate_rewards)
                evaluations += evalu
                break
            immediate_rewards.append(reward_stillplaying)

        if training_episode % update_frequency == 0:
            logger.info("Average number of training steps: %f", len(evaluations) / update_frequency)
            neuro.update_gradients(*zip(*evaluations))
            evaluations = []


if __name__ == "__main__":
    main()

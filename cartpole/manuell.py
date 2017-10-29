from itertools import count
import random

import gym
import numpy as np
import tensorflow as tf

def Neuro():
    n_in = 4
    n_hidden = 3
    n_out = 2
    initializer = tf.contrib.layers.variance_scaling_initializer()
    input_ = tf.placeholder(dtype=tf.float32, shape=[None, n_in])
    hidden = tf.layers.dense(input_, n_hidden, activation=tf.nn.sigmoid)
    output = tf.layers.dense(hidden, 2)
    probs = tf.nn.softmax(output)
    return input_, probs


def Optimizer():
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate)


def update_net(net, factor, opt):
    grads_and_vars = [(factor * sess.run(grad, feed_dict={input_: test_val}), var) 
                      for grad, var in opt.compute_gradients(net)]
    opt.apply_gradients(grads_and_vars)


def TrainingOp(optimizer, grad_var_list):
    grad_placeholders = [(tf.placeholder(tf.float32, shape=grad.get_shape()), var)
                         for grad, var in grad_var_list]
    return optimizer.apply_gradients(grad_placeholders)


def calculate_mean_gradient(states, actions, input_, dp0dW):
    grads = [g.eval(feed_dict={input_: np.vstack(states[:-1])}) for g, v in dp0dW]
    import ipdb; ipdb.set_trace()


def main():
    input_, neuro = Neuro()
    optimizer = Optimizer()
    init = tf.global_variables_initializer()    

    env = gym.make('CartPole-v0')

    dp0dW = optimizer.compute_gradients(neuro[0, 0])
    dp1dW = optimizer.compute_gradients(neuro[0, 1])
    training_op = TrainingOp(optimizer, dp0dW)

    sess = tf.InteractiveSession()
    init.run()
    for training_episode in range(10000):
        obs = env.reset()
        states, actions = [obs], []

        for training_step in range(1000):
            env.render()
            pos, vel, angle, angle_vel = obs
            #  step = 1 if angle_vel >= 0 else 0
            step = random.randrange(0, 2)
            actions.append(step)
            obs, reward, done, info = env.step(step)
            states.append(obs)
            if done:
                import ipdb; ipdb.set_trace()
                grads_per_state = [
                                   [g.eval(feed_dict={input_: state[None, :]}) for g, v in dp0dW]
                                   for state in states
                                  ]
                calculate_mean_gradient(states, actions, input_, grads_per_state)
                break


if __name__ == "__main__":
    main()

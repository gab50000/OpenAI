from itertools import count
import random

import gym
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


def main():
    input_, neuro = Neuro()
    optimizer = Optimizer()
    init = tf.global_variables_initializer()    
    sess = tf.InteractiveSession()
    init.run()

    env = gym.make('CartPole-v0')

    for _ in range(1000):
        for training_step in range(100):
            obs = env.reset()
            for i in count():
                #  env.render()
                pos, vel, angle, angle_vel = obs
                #  step = 1 if angle_vel >= 0 else 0
                step = random.randrange(0, 2)
                obs, reward, done, info = env.step(step)
                print(sess.run(neuro, feed_dict={input_: obs[None, :]}))
                if done:
                    if i > 100:
                        print(i)
                    break


if __name__ == "__main__":
    main()

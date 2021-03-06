"""Playground for basic Tensorflow funtionality"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def constants():
    session = tf.InteractiveSession()
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicitly
    print(session.run([node1, node2]))


def sigmoid():
    hidden_size = 10
    mean_pos = 0
    session = tf.InteractiveSession()
    input_node = tf.placeholder(tf.float32, name="input", shape=[None, 1])
    W1 = tf.Variable(tf.random_normal([1, hidden_size], mean=mean_pos), name="W1")
    b1 = tf.Variable(tf.random_normal([hidden_size], mean=mean_pos), name="b1")
    W2 = tf.Variable(tf.random_normal([hidden_size, 1], mean=mean_pos), name="W2")
    b2 = tf.Variable(tf.random_normal([1], mean=mean_pos), name="b2")
    hidden_node = tf.nn.tanh(tf.matmul(input_node, W1) + b1, name="hidden")
    output_node = tf.reduce_sum(tf.matmul(hidden_node, W2) + b2, axis=-1, keep_dims=True)
    y = tf.placeholder(tf.float32, shape=[None, 1])

    writer = tf.summary.FileWriter(".", session.graph)
    init = tf.global_variables_initializer()
    session.run(init)
    print(session.run(output_node, {input_node: [[5]]}))
    writer.close()

    x_train = np.linspace(-5, 5).reshape((-1, 1))
    y_train = np.sin(x_train)

    loss = tf.reduce_sum(tf.square(output_node - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)

    for i in range(1000):
        if i % 100 == 0:
            print(f"Iteration {i:3d}", end="\r")
        session.run(train, {input_node: x_train, y: y_train})

    net_out = session.run(output_node, {input_node: x_train})

    w1, w2 = session.run([W1, W2])

    if np.isnan(w1).any() or np.isnan(w2).any():
        if np.isnan(w1).all() or np.isnan(w2).all():
            raise ValueError("All weights are None")
        else:
            raise ValueError("Some weights are None")

        exit()
    plt.plot(x_train, y_train, "x")
    plt.plot(x_train, net_out)
    plt.show()


if __name__ == "__main__":
    sigmoid()

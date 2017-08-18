"""Playground for basic Tensorflow funtionality"""
import tensorflow as tf


def constants():
    session = tf.InteractiveSession()
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicitly
    print(session.run([node1, node2]))


def sigmoid():
    session = tf.InteractiveSession()
    input_node = tf.placeholder(tf.float32)
    W = tf.constant(4.0)
    output_node = W * input_node
    writer = tf.summary.FileWriter(".", session.graph)
    print(session.run(output_node, {input_node: 5}))
    writer.close()


if __name__ == "__main__":
    sigmoid()

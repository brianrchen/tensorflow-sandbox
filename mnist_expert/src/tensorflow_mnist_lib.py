import numpy as np
import sys
import tensorflow as tf

def load_data():
    # How to cache these results? Maybe they are already, I just moved directories
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets('MNIST_data', one_hot=True)

def train_softmax():
    # not sure this needs to be abstracted
    # huh, if you don't put () on load_data in this case, you only catch the
    # error when you try to call .train, and find that the function doesn't
    # have that field. Would this be caught by an IDE/type hinting?
    mnist = load_data()

    # this is probably the wrong thing to do; figure out sessions concept
    sess = tf.InteractiveSession()

    # placeholders seem to be "placeholders" for data to load into the network;
    # at "run-time" for the computation graph, you can feed data or values in
    # via the placeholders
    # If you had some kind of need for hot-loading data, you can use a placeholder
    # "None" indicates that the first dimension, corresponding to the batch size,
    # can be any size.  I wonder if this would've been better as an enum?  special
    # case values are a little magical.
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # A Variable is used to hold values that live in a tensorflow graph, and
    # can be used and modified by tensorflow. You could use inputs and outputs
    # for the weights for every iteration, but that would involve a lot of
    # boilerplate.
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    # Most logic in tensorflow seems to be about setting up a graph, so values
    # are not set until explicitly set. Variables must be initialized to actually
    # have the value you "set" them with beforehand.  This allows you to defer
    # computation until the graph is set up properly.  Not sure about the possible
    # alternatives in this design, or if I understand the benefits well enough.
    sess.run(tf.initialize_all_variables())

    # in this case y=tf.nn.softmax specifies a graph construction.  Want to break
    # down different concepts in tensorflow to understand what pieces are at play,
    # what it looks like to construct a tensorflow model, what tools are available.
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # so one of the concepts seems to be specifying optimization steps as objects?
    # Want to find docs about how the concepts are composed.
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def main(args):
    pass

if __name__ == "__main__":
    main(sys.argv)

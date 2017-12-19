"""My modified file for the ./2_fullyconnected.ipynb file."""

# coding: utf-8

# pylama: ignore=E402,D103

# Deep Learning
# =============
#
# Assignment 2
# ------------
#
# Previously in `1_notmnist.ipynb`, we created a pickle with formatted datasets
# for training, development and testing on the [notMNIST
# dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).
#
# The goal of this assignment is to progressively train deeper and more
# accurate models using TensorFlow.


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

import google_course_library as gcl

# First reload the data we generated in `1_notmnist.ipynb`.


with open(gcl.DATASET_PICKLE_FILE, 'rb') as f:
    save = pickle.load(f)
    x_train = save['train_dataset']
    y_train = save['train_labels']
    x_validation = save['valid_dataset']
    y_validation = save['valid_labels']
    x_test = save['test_dataset']
    y_test = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', x_train.shape, y_train.shape)
    print('Validation set', x_validation.shape, y_validation.shape)
    print('Test set', x_test.shape, y_test.shape)

# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.
IMAGE_SIZE = 28
NUM_LABELS = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


x_train, y_train = reformat(x_train, y_train)
x_validation, y_validation = reformat(x_validation, y_validation)
x_test, y_test = reformat(x_test, y_test)
print('Training set', x_train.shape, y_train.shape)
print('Validation set', x_validation.shape, y_validation.shape)
print('Test set', x_test.shape, y_test.shape)


# Removed the logistic regression using GD.

# Let's now switch to stochastic gradient descent training instead, which is
# much faster.
#
# The graph will be similar, except that instead of holding all the training
# data into a constant node, we create a `Placeholder` node which will be fed
# actual data at every call of `session.run()`.


batch_size = 128

graph = tf.Graph()
N_NEURONS = 2 * 1024
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(
            batch_size, IMAGE_SIZE * IMAGE_SIZE))
    tf_train_labels = tf.placeholder(
        tf.float32, shape=(
            batch_size, NUM_LABELS))
    tf_valid_dataset = tf.constant(x_validation)
    tf_test_dataset = tf.constant(x_test)

    # Variables.
    first_weights = tf.Variable(
        tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, N_NEURONS]))
    biases = tf.Variable(tf.zeros([N_NEURONS]))

    # Training computation.
    dot_sum = tf.matmul(tf_train_dataset, first_weights) + biases
    # Add the relu operation.
    apply_relu = tf.nn.relu(dot_sum)

    # Add second weights
    second_weights = tf.Variable(
        tf.truncated_normal([N_NEURONS, NUM_LABELS]))
    apply_resize = tf.matmul(apply_relu, second_weights)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=tf_train_labels,
            logits=apply_resize))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(apply_resize)
    # Prediction for test.
    dot_sum_test = tf.matmul(tf_test_dataset, first_weights) + biases
    apply_relu_test = tf.nn.relu(dot_sum_test)
    apply_resize_test = tf.matmul(apply_relu_test, second_weights)
    test_prediction = tf.nn.softmax(apply_resize_test)
    # Prediction for validation.
    dot_sum_valid = tf.matmul(tf_valid_dataset, first_weights) + biases
    apply_relu_valid = tf.nn.relu(dot_sum_valid)
    apply_resize_valid = tf.matmul(apply_relu_valid, second_weights)
    valid_prediction = tf.nn.softmax(apply_resize_valid)

# Let's run it:


num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = x_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be
        # fed, and the value is the numpy array to feed to it.
        feed_dict = {
            tf_train_dataset: batch_data,
            tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print(
                "Minibatch accuracy: %.1f%%" %
                accuracy(
                    predictions,
                    batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), y_validation))
    print(
        "Test accuracy: %.1f%%" %
        accuracy(
            test_prediction.eval(),
            y_test))

# ---
# Problem
# -------
#
# Turn the logistic regression example with SGD into a 1-hidden layer neural
# network with rectified linear units
# [nn.relu()](https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#relu)
# and 1024 hidden nodes. This model should improve your validation / test
# accuracy.
#
# ---

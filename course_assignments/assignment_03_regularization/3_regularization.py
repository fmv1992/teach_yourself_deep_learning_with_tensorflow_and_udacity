"""My modified file for the ./3_regularization.ipynb file."""

# pylama: ignore=E402,D103

# Deep Learning
# =============
#
# Assignment 3
# ------------
#
# Previously in `2_fullyconnected.ipynb`, you trained a logistic regression and
# a neural network model.
#
# The goal of this assignment is to explore regularization techniques.


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

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
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return dataset, labels


x_train, y_train = reformat(x_train, y_train)
x_validation, y_validation = reformat(x_validation, y_validation)
x_test, y_test = reformat(x_test, y_test)
print('Training set', x_train.shape, y_train.shape)
print('Validation set', x_validation.shape, y_validation.shape)
print('Test set', x_test.shape, y_test.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


# PROBLEM 1
# ---
# Problem 1
# ---------
#
# Introduce and tune L2 regularization for both logistic and neural network
# models. Remember that L2 amounts to adding a penalty on the norm of the
# weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor
# `t` using `nn.l2_loss(t)`. The right amount of regularization should improve
# your validation / test accuracy.
#
# ---

# DROPOUT_KEEP_CHANCE = 1
BATCH_SIZE = 128
N_NEURONS = 1024
TRAINING_STEPS = 31
REGULARIZATION_SCALING = float(1e-3)
LOGGING_CYCLE = 10

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    is_training = tf.placeholder(tf.bool)
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(
            BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE))
    tf_train_labels = tf.placeholder(
        tf.float32, shape=(
            BATCH_SIZE, NUM_LABELS))
    tf_valid_dataset = tf.constant(x_validation)
    tf_test_dataset = tf.constant(x_test)

    # Variables.
    DROPOUT_KEEP_CHANCE = tf.cond(is_training,
                                  lambda: tf.Variable(0.5),
                                  lambda: tf.Variable(1.))
    first_weights = tf.Variable(
        tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, N_NEURONS]))
    first_weights = tf.nn.dropout(first_weights, DROPOUT_KEEP_CHANCE)

    first_biases = tf.Variable(tf.zeros([N_NEURONS]))

    # Training computation.
    dot_sum = tf.matmul(tf_train_dataset, first_weights) + first_biases
    # Add the relu operation.
    apply_relu = tf.nn.relu(dot_sum)

    # Add second weights
    second_weights = tf.Variable(
        tf.truncated_normal([N_NEURONS, NUM_LABELS]))
    second_weights = tf.nn.dropout(second_weights, DROPOUT_KEEP_CHANCE)
    second_biases = tf.Variable(tf.zeros([NUM_LABELS]))
    apply_resize = tf.matmul(apply_relu, second_weights) + second_biases

    # Calculate regularization term.
    reg_l2 = tf.constant(REGULARIZATION_SCALING) * (
        tf.nn.l2_loss(first_weights) + tf.nn.l2_loss(second_weights))

    loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf_train_labels, logits=apply_resize))
            + tf.reduce_mean(reg_l2))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(apply_resize)
    # Prediction for test.
    dot_sum_test = tf.matmul(tf_test_dataset, first_weights) + first_biases
    apply_relu_test = tf.nn.relu(dot_sum_test)
    apply_resize_test = tf.matmul(apply_relu_test, second_weights)
    test_prediction = tf.nn.softmax(apply_resize_test)
    # Prediction for validation.
    dot_sum_valid = tf.matmul(tf_valid_dataset, first_weights) + first_biases
    apply_relu_valid = tf.nn.relu(dot_sum_valid)
    apply_resize_valid = tf.matmul(apply_relu_valid, second_weights)
    valid_prediction = tf.nn.softmax(apply_resize_valid)

# Let's run it:


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(TRAINING_STEPS):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (y_train.shape[0] - BATCH_SIZE)
        # Generate a minibatch.
        batch_data = x_train[offset:(offset + BATCH_SIZE), :]
        batch_labels = y_train[offset:(offset + BATCH_SIZE), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be
        # fed, and the value is the numpy array to feed to it.
        training_feed_dict = {
            is_training: True,
            tf_train_dataset: batch_data,
            tf_train_labels: batch_labels}
        test_feed_dict = {is_training: False}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction],
            feed_dict=training_feed_dict)
        if (step % LOGGING_CYCLE == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions,
                                                          batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(feed_dict=test_feed_dict), y_validation))
    print(
        "Test accuracy: %.1f%%" %
        accuracy(
            test_prediction.eval(feed_dict=test_feed_dict),
            y_test))


# PROBLEM 2
# ---
# Problem 2
# ---------
# Let's demonstrate an extreme case of overfitting. Restrict your training data
# to just a few batches. What happens?
#
# ---

# PROBLEM 3
# ---
# Problem 3
# ---------
# Introduce Dropout on the hidden layer of the neural network. Remember:
# Dropout should only be introduced during training, not evaluation, otherwise
# your evaluation results would be stochastic as well. TensorFlow provides
# `nn.dropout()` for that, but you have to make sure it's only inserted during
# training.
#
# What happens to our extreme overfitting case?
#
# ---

# PROBLEM 4
# ---
# Problem 4
# ---------
#
# Try to get the best performance you can using a multi-layer model! The best
# reported test accuracy using a deep network is
# [97.1%](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595).
#
# One avenue you can explore is to add multiple layers.
#
# Another one is to use learning rate decay:
#
#     global_step = tf.Variable(0)  # count the number of steps taken.
#     learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step) # noqa
#  ---
#

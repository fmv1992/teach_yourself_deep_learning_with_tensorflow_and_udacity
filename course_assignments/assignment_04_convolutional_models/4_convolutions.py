"""My modified file for the ./3_regularization.ipynb file."""

# pylama: ignore=E402,D103

# Deep Learning
# =============
#
# Assignment 4
# ------------
#
# Previously in `2_fullyconnected.ipynb` and `3_regularization.ipynb`, we
# trained fully connected networks to classify
# [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)
# characters.
#
# The goal of this assignment is make the neural network convolutional.


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


import google_course_library as gcl


def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, IMAGE_SIZE, IMAGE_SIZE, num_channels)).astype(np.float32)
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


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


# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by
# #channels)
# - labels as float 1-hot encodings.


IMAGE_SIZE = 28
NUM_LABELS = 10
num_channels = 1  # grayscale


x_train, y_train = reformat(x_train, y_train)
x_validation, y_validation = reformat(x_validation, y_validation)
x_test, y_test = reformat(x_test, y_test)
print('Training set', x_train.shape, y_train.shape)
print('Validation set', x_validation.shape, y_validation.shape)
print('Test set', x_test.shape, y_test.shape)


# Let's build a small network with two convolutional layers, followed by
# one fully connected layer. Convolutional networks are more expensive
# computationally, so we'll limit its depth and number of fully connected
# nodes.


batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
STRIDE_SIZE = 2

graph = tf.Graph()

with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, num_channels))
    tf_train_labels = tf.placeholder(
        tf.float32, shape=(
            batch_size, NUM_LABELS))
    tf_valid_dataset = tf.constant(x_validation)
    tf_test_dataset = tf.constant(x_test)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [(IMAGE_SIZE // 4) * (IMAGE_SIZE // 4) * depth,
         num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, NUM_LABELS], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(
            hidden, layer2_weights, [
                1, STRIDE_SIZE, STRIDE_SIZE, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(
            hidden, [
                shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=tf_train_labels,
            logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


num_steps = 1001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        batch_data = x_train[offset:(offset + batch_size), :, :, :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        feed_dict = {
            tf_train_dataset: batch_data,
            tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print(
                'Minibatch accuracy: %.1f%%' %
                accuracy(
                    predictions,
                    batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), y_validation))
    print(
        'Test accuracy: %.1f%%' %
        accuracy(
            test_prediction.eval(),
            y_test))


# ---
# Problem 1
# ---------
#
# The convolutional model above uses convolutions with stride 2 to reduce the
# dimensionality. Replace the strides by a max pooling operation
# (`nn.max_pool()`) of stride 2 and kernel size 2.
#
# ---

# ---
# Problem 2
# ---------
#
# Try to get the best performance you can using a convolutional net. Look for
# example at the classic [LeNet5](http://yann.lecun.com/exdb/lenet/)
# architecture, adding Dropout, and/or adding learning rate decay.
#
# ---

# Problem 1

*Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor `t` using `nn.l2_loss(t)`. The right amount of regularization should improve your validation / test accuracy.*

See the code. I was uncertain on how to put multiple weight matrix into the regularization term. It turns out that one has just to add them:

~~~~~~
reg_l2 = tf.constant(REGULARIZATION_SCALING) * (
    tf.nn.l2_loss(first_weights)
    + tf.nn.l2_loss(third_weights)
    [...]
    + tf.nn.l2_loss(second_weights)
~~~~~~

# Problem 2

*Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?*

Changing `TRAINING_STEPS = 3001` to `TRAINING_STEPS = 100`. one goes from:

    Minibatch loss at step 0: 679.978394
    Minibatch accuracy: 7.0%
    Validation accuracy: 20.4%
    Minibatch loss at step 500: 193.288696
    Minibatch accuracy: 80.5%
    Validation accuracy: 79.9%
    Minibatch loss at step 1000: 115.152863
    Minibatch accuracy: 81.2%
    Validation accuracy: 82.1%
    Minibatch loss at step 1500: 68.540474
    Minibatch accuracy: 89.8%
    Validation accuracy: 83.3%
    Minibatch loss at step 2000: 41.630451
    Minibatch accuracy: 82.8%
    Validation accuracy: 85.4%
    Minibatch loss at step 2500: 25.228914
    Minibatch accuracy: 89.1%
    Validation accuracy: 86.1%
    Minibatch loss at step 3000: 15.356815
    Minibatch accuracy: 93.0%
    Validation accuracy: 87.1%
    Test accuracy: 93.0%

To:

    Minibatch loss at step 0: 619.013428
    Minibatch accuracy: 8.6%
    Validation accuracy: 38.0%
    Minibatch loss at step 10: 390.292419
    Minibatch accuracy: 73.4%
    Validation accuracy: 74.8%
    Minibatch loss at step 20: 351.228638
    Minibatch accuracy: 75.0%
    Validation accuracy: 78.2%
    Minibatch loss at step 30: 353.483215
    Minibatch accuracy: 76.6%
    Validation accuracy: 77.4%
    Minibatch loss at step 40: 349.004028
    Minibatch accuracy: 74.2%
    Validation accuracy: 78.5%
    Minibatch loss at step 50: 347.874817
    Minibatch accuracy: 77.3%
    Validation accuracy: 78.4%
    Minibatch loss at step 60: 328.809143
    Minibatch accuracy: 78.1%
    Validation accuracy: 76.7%
    Minibatch loss at step 70: 319.431671
    Minibatch accuracy: 82.0%
    Validation accuracy: 79.0%
    Minibatch loss at step 80: 319.949097
    Minibatch accuracy: 79.7%
    Validation accuracy: 78.8%
    Minibatch loss at step 90: 304.265564
    Minibatch accuracy: 81.2%
    Validation accuracy: 78.6%
    Test accuracy: 86.0%

What happens is that the model does not train to its best. The weight values are "underdeveloped" and not correctly fit to the problem.

# Problem 3
# Problem 4

http://neuralnetworksanddeeplearning.com/chap3.html

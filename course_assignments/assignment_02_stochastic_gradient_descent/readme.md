# Problem

*Turn the logistic regression example with SGD into a 1-hidden layer neural network with rectified linear units [nn.relu()](https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#relu) and 1024 hidden nodes. This model should improve your validation / test accuracy.*

To improve my understanding of TensorFlow I have read the following tutorial:

1. https://www.tensorflow.org/get_started/get_started

    Then:

1. https://www.tensorflow.org/get_started/mnist/pros



# Results

The Artificial Neural Networks (referred to as "ANN" onwards) got a better result compared to the [Logistic Regression Assignment](https://github.com/fmv1992/teach_yourself_deep_learning_with_tensorflow_and_udacity/tree/dev/course_assignments/assignment_01_notmnist):

    Minibatch loss at step 0: 546.684082
    Minibatch accuracy: 6.2%
    Validation accuracy: 43.8%
    Minibatch loss at step 500: 26.489553
    Minibatch accuracy: 78.9%
    Validation accuracy: 82.7%
    Minibatch loss at step 1000: 13.505351
    Minibatch accuracy: 84.4%
    Validation accuracy: 81.8%
    Minibatch loss at step 1500: 4.903067
    Minibatch accuracy: 90.6%
    Validation accuracy: 83.6%
    Minibatch loss at step 2000: 8.811088
    Minibatch accuracy: 84.4%
    Validation accuracy: 84.0%
    Minibatch loss at step 2500: 5.224963
    Minibatch accuracy: 90.6%
    Validation accuracy: 83.8%
    Minibatch loss at step 3000: 2.340058
    Minibatch accuracy: 91.4%
    Validation accuracy: 84.4%
    **Test accuracy: 90.9%**

An improvement of accuracy of **7.7%**. (commit `ffbfcc1c7f1965d39ba101be8aec63a19edfe617`).

I have purposefully omitted the bias in when feeding it to the output layer. With that included the new accuracy is:

    Test accuracy: 90.5%

Which is not an improvement :)

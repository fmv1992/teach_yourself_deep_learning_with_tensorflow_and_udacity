I had a lot of trouble getting the matrix sizes right when moving from a convolutional network to a maxpooling network.

In the convolutional network the architecture is the following:


~~~~~~
[batch_size, 28*28] |
                    |
input layer         |
~~~~~~

# Shapes and forms

## For both convolution and maxpooling

1. Image size.
    1. Images shapes: the `reformat` function makes each image `(28, 28, 1)`.
    1. You sample the image with same padding so that after the padding the shape is unchanged.

1. Patch size.
    1. Each patch size is of the the size `(patch_size, patch_size, num_channels, depth)`.

## For convolution

1. Convoluted image.
    1. From the documentation of `tf.nn.conv2d`

            Returns:

            A Tensor. Has the same type as input. A 4-D tensor. The dimension order is determined by the value of data_format, see below for details.

        So in our case it has the same shape as the data: `(batch_size, 28, 28, 1)`.

1. Inside the convolution `model` function:

        Training set (100000, 28, 28) (100000,)
        Validation set (10000, 28, 28) (10000,)
        Test set (10000, 28, 28) (10000,)
        Training set (100000, 28, 28, 1) (100000, 10)
        Validation set (10000, 28, 28, 1) (10000, 10)
        Test set (10000, 28, 28, 1) (10000, 10)

        layer1 (5, 5, 1, 16)
        layer2 (5, 5, 16, 16)
        layer3 (784, 64)
        layer4 (64, 10)

        (convolution function)
        data (16, 28, 28, 1)
        hid1 (16, 14, 14, 16)
        conv1 (16, 14, 14, 16)
        hid2 (16, 7, 7, 16)
        conv2 (16, 7, 7, 16)
        reshape (16, 784)

## For maxpooling

1. Pooled image.

        TODO

1. Inside the maxpooling `model` function:

        TODO

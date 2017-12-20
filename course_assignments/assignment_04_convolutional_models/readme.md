# Problem 1

I had a lot of trouble getting the matrix sizes right when moving from a convolutional network to a maxpooling network.

To change between convolution and maxpooling change the following variable:

    # Used to switch between convolution and maxpooling.
    is_convolution = False
    is_convolution = True


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
        layer3 (64, 64)
        layer4 (64, 10)
        data (16, 28, 28, 1)
        hid1 (16, 6, 6, 16)
        conv1 (16, 6, 6, 16)
        hid2 (16, 2, 2, 16)
        conv2 (16, 2, 2, 16)
        reshape (16, 64)
        data (10000, 28, 28, 1)
        hid1 (10000, 6, 6, 16)
        conv1 (10000, 6, 6, 16)
        hid2 (10000, 2, 2, 16)
        conv2 (10000, 2, 2, 16)
        reshape (10000, 64)
        data (10000, 28, 28, 1)
        hid1 (10000, 6, 6, 16)
        conv1 (10000, 6, 6, 16)
        hid2 (10000, 2, 2, 16)
        conv2 (10000, 2, 2, 16)
        reshape (10000, 64)


## For maxpooling

1. Inside the maxpooling `model` function:

        Training set (100000, 28, 28) (100000,)
        Validation set (10000, 28, 28) (10000,)
        Test set (10000, 28, 28) (10000,)
        Training set (100000, 28, 28, 1) (100000, 10)
        Validation set (10000, 28, 28, 1) (10000, 10)
        Test set (10000, 28, 28, 1) (10000, 10)
        layer1 (2, 2, 1, 16)
        layer2 (2, 2, 16, 16)
        layer3 (256, 64)
        layer4 (64, 10)
        data (16, 28, 28, 1)
        maxp1 (16, 4, 4, 1)
        hid1 (16, 4, 4, 16)
        [16, 4, 4, 16]
        reshape (16, 256)
        hid3 (16, 64)
        data (10000, 28, 28, 1)
        maxp1 (10000, 4, 4, 1)
        hid1 (10000, 4, 4, 16)
        [10000, 4, 4, 16]
        reshape (10000, 256)
        hid3 (10000, 64)
        data (10000, 28, 28, 1)
        maxp1 (10000, 4, 4, 1)
        hid1 (10000, 4, 4, 16)
        [10000, 4, 4, 16]
        reshape (10000, 256)
        hid3 (10000, 64)

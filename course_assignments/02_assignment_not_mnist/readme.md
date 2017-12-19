# Problem 1

I have loaded the images with a 1 second timer. We can see that some images are very unreadable such as a one-pixel-wide-letter-i. There are other images which look like animals (a penguin for instance) or symbols (like an yin-yang symbol).

# Problem 2

Images still look good.

# Problem 3

The data is balanced:

    a class has 52909 elements.
    b class has 52911 elements.
    c class has 52912 elements.
    d class has 52911 elements.
    e class has 52912 elements.
    f class has 52912 elements.
    g class has 52912 elements.
    h class has 52912 elements.
    i class has 52912 elements.
    j class has 52911 elements.

# Problem 4

Images still look good.

# Problem 5

The code output shows that the image groups have little intersection with each other.
To do that a combination of the (train, test, validation) was combined into groups of 1 by 1, 2 by 2 and 3 by 3.

This first section should be compared with the size of each group:

    intersection ('train',): has 96111 unique elements out of 96111 total.
    intersection ('test',): has 9808 unique elements out of 9808 total.
    intersection ('validation',): has 9861 unique elements out of 9861 total.

Considering that each group size has:

    TRAIN_SIZE = 100000
    VALID_SIZE = 10000
    TEST_SIZE = 10000

Elements. We have at most 4% of duplicates then. This number is even lower if groups are combined:


    intersection ('train', 'test'): has 105346 unique elements out of 105919 total.
    intersection ('train', 'validation'): has 105465 unique elements out of 105972 total.
    intersection ('test', 'validation'): has 19605 unique elements out of 19669 total.
    intersection ('train', 'test', 'validation'): has 114645 unique elements out of 115780 total.

# Problem 6

A logistic model was trained with grid search for hyperparameters, achieving significant results.

    Train logit score: 83.38%.
    Test logit score: 83.05%.

"""My modified file for the ./1_notmnist.ipynb file."""

# pylama: ignore=E402,D103

# Deep Learning =============
#
# Assignment 1 ------------
#
# The objective of this assignment is to learn about simple data curation
# practices, and familiarize you with some of the data we'll be reusing later.
#
# This notebook uses the
# [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)
# dataset to be used with python experiments. This dataset is designed to look
# like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while
# looking a little more like real data: it's a harder task, and the data is a
# lot less 'clean' than MNIST.

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function

import matplotlib
try:
    matplotlib.use('Qt5Agg')
except ValueError:
    matplotlib.use('Qt4Agg')

import glob
import itertools
import os
import random
import sys
import tarfile

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import imageio
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

import google_course_library as gcl

import logit_classifier

# Define program constants.
gcl.train_size = 10000
gcl.validation_size = 1000
gcl.test_size = 1000

# Config the matplotlib backend as plotting inline in IPython
# get_ipython().magic('matplotlib inline')

# First, we'll download the dataset to our local machine. The data consists of
# characters rendered in a variety of fonts on a 28x28 image. The labels are
# limited to 'A' through 'J' (10 classes). The training set has about 500k and
# the testset 19000 labeled examples. Given these sizes, it should be possible
# to train models quickly on any machine.

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None


def download_progress_hook(count, blockSize, totalSize):
    """Report the progress of a download.

    This is mostly intended for users with slow internet connections. Reports
    every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(gcl.DATA_PATH, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(
            url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' +
            dest_filename +
            '. Can you get to it with a browser?')
    return dest_filename


train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

# Extract the dataset from the compressed .tar.gz file.
# This should give you a set of directories, labeled A through J.

num_classes = 10
np.random.seed(0)
random.seed(0)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        # print(
        #     '%s already present - Skipping extraction of %s.' %
        #     (root, filename))
        pass
    else:
        print(
            'Extracting data for %s. This may take a while. Please wait.' %
            root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(gcl.DATA_PATH)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    # print(data_folders)
    return data_folders


train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

# PROBLEM 1
# --- Problem 1 ---------
#
# Let's take a peek at some of the data to make sure it looks sensible. Each
# exemplar should be an image of a character A through J rendered in a
# different font. Display a sample of the images that we just downloaded. Hint:
# you can use the package IPython.display.
#
# ---
all_png_files = glob.glob(os.path.join(gcl.DATA_PATH, '**' + os.sep + '*.png'),
                          recursive=True)
for png_f in random.sample(all_png_files, 10):
    fig = plt.figure()
    timer = fig.canvas.new_timer(interval=1000)
    timer.add_callback(plt.close)
    img = mpimg.imread(png_f)
    timer.start()
    plt.imshow(img)
    plt.show()
    del img, timer, fig
plt.close('all')

# Now let's load the data in a more manageable format. Since, depending on your
# computer setup you might not be able to fit it all in memory, we'll load each
# class into a separate dataset, store them on disk and curate them
# independently. Later we'll merge them into a single dataset of manageable
# size.
#
# We'll convert the entire dataset into a 3D array (image index, x, y) of
# floating point values, normalized to have approximately zero mean and
# standard deviation ~0.5 to make training easier down the road.
#
# A few images might not be readable, we'll just skip them.

IMAGE_SIZE = 28  # Pixel width and height.
PIXEL_DEPTH = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), IMAGE_SIZE, IMAGE_SIZE),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (imageio.imread(image_file).astype(float) -
                          PIXEL_DEPTH / 2) / PIXEL_DEPTH
            if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise Exception('Unexpected image shape: %s' %
                                str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except (IOError, ValueError) as e:
            print(
                'Could not read:',
                image_file,
                ':',
                e,
                '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

# PROBLEM 2
# --- Problem 2 ---------
#
# Let's verify that the data still looks good. Displaying a sample of the
# labels and images from the ndarray. Hint: you can use matplotlib.pyplot.
#
# ---
train_dict = dict()
for train_letter in train_datasets:
    letter = train_letter.strip('.pickle')[-1].lower()
    with open(train_letter, 'rb') as f:
        train_dict[letter] = pickle.load(f)

keys = np.random.choice(list(train_dict.keys()), size=10, replace=True)
for key in keys:
    letter_array = train_dict[key]
    gcl.plot_timed_sample_from_array(letter_array, n=1)

# PROBLEM 3
# --- Problem 3 --------- Another check: we expect the data to be balanced
# across classes. Verify that.
#
# ---


def check_dictionary_is_balanced(mydict):
    for one_class in mydict.keys():
        print('{0} class has {1} elements.'.format(one_class,
                                                   len(mydict[one_class])))


check_dictionary_is_balanced(train_dict)

# Merge and prune the training data as needed. Depending on your computer
# setup, you might not be able to fit it all in memory, and you can tune
# `gcl.train_size` as needed. The labels will be stored into a separate array
# of integers 0 through 9.
#
# Also create a validation dataset for hyperparameter tuning.


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, IMAGE_SIZE)
    train_dataset, train_labels = make_arrays(train_size, IMAGE_SIZE)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and
                # training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels



valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, gcl.train_size, gcl.validation_size)
_, _, test_dataset, test_labels = merge_datasets(
    test_datasets, gcl.test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

# Next, we'll randomize the data. It's important to have the labels well
# shuffled for the training and test distributions to match.


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

# ---
# Problem 4
# ---------
# Convince yourself that the data is still good after shuffling!
#
# ---

for arr in (train_dataset, test_dataset, valid_dataset):
    gcl.plot_timed_sample_from_array(arr, n=3)


try:
    f = open(gcl.DATASET_PICKLE_FILE, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', gcl.DATASET_PICKLE_FILE, ':', e)
    raise

statinfo = os.stat(gcl.DATASET_PICKLE_FILE)
print('Compressed pickle size:', statinfo.st_size)

# --- Problem 5 ---------
#
# By construction, this dataset might contain a lot of overlapping samples,
# including training data that's also contained in the validation and test set!
# Overlap between training and test can skew the results if you expect to use
# your model in an environment where there is never an overlap, but are
# actually ok if you expect to see training samples recur when you use it.
# Measure how much overlap there is between training, validation and test
# samples.
#
# Optional questions: - What about near duplicates between datasets? (images
# that are almost identical) - Create a sanitized validation and test set, and
# compare your accuracy on those in subsequent assignments. ---
validation_dataset = valid_dataset

train_unique = np.unique(train_dataset, axis=0)
validation_unique = np.unique(validation_dataset, axis=0)
test_unique = np.unique(test_dataset, axis=0)

deduped_set = {'train': train_unique,
               'test': test_unique,
               'validation': validation_unique}
for combination in (1, 2, 3):
    for keys in itertools.combinations(list(deduped_set.keys()), combination):

        datasets = [deduped_set[k] for k in keys]
        total_elements = sum(map(lambda x: x.shape[0], datasets))
        unique_intersect = np.unique(np.concatenate(datasets), axis=0).shape[0]

        print('intersection {0}: has {1} unique elements out of {2} '
              'total.'.format(keys, unique_intersect, total_elements))

# train_list = np.split(train_dataset, train_dataset.shape[0])
# train_list = [np.vstack(x) for x in train_list]
# train_unique = np.unique(train_dataset)
#
#
# validation_list = np.split(validation_dataset, validation_dataset.shape[0])
# validation_list = [np.vstack(x) for x in validation_list]
# validation_set = set(validation_list)
# import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

# --- Problem 6 ---------
#
# Let's get an idea of what an off-the-shelf classifier can give you on this
# data. It's always good to check that there is something to learn, and that
# it's a problem that is not so trivial that a canned solution solves it.
#
# Train a simple model on this data using 50, 100, 1000 and 5000 training
# samples. Hint: you can use the LogisticRegression model from
# sklearn.linear_model.
#
# Optional question: train an off-the-shelf model on all the data!
#
# ---

# You may want to change the gcl.train_size, gcl.validation_size, gcl.test_size constants to
# train faster and check code correctness.

logit_classifier.main(train_dataset, valid_dataset, train_labels, valid_labels)

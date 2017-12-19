"""Classifier part for the first assignment."""

import pickle
import os

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV

import google_course_library as gcl

OUT_PATH = os.path.join(gcl.DATA_PATH, '02_assignment_not_mnist')
MODEL_PATH = os.path.join(OUT_PATH, 'model.pickle')

# Old grid search parameters before changing to LogisticRegressionCV.
# PARAM_GRID = [
#     {'penalty': ['l2', ],
#      'C': [0.01, 0.1, 1., 10],
#      },
# ]
LRCV_PARAMS = {
    'random_state': 0,
    'multi_class': 'multinomial',
    'solver': 'sag',
    'max_iter': 5000,
    'class_weight': 'balanced',
    'n_jobs': -2,  # Use all but one core.
    'verbose': 3
}


def train_classifier(x_train, y_train):
    """Train a logistic classifier. Return the classifier."""
    if not os.path.exists(MODEL_PATH):
        clf = LogisticRegressionCV(**LRCV_PARAMS)
        clf.fit(x_train, y_train)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(clf, f)
    else:
        with open(MODEL_PATH, 'rb') as f:
            clf = pickle.load(f)
    return clf


def report_model(classifier, x_train, x_test, y_train, y_test):
    """Report model results' metrics."""
    print('Train logit score: {0:2.2%}.'.format(classifier.score(x_train,
                                                                 y_train)))
    print('Test logit score: {0:2.2%}.'.format(classifier.score(x_test,
                                                                y_test)))


def main(x_train, x_test, y_train, y_test):
    """Train and report model (with persistence)."""
    x_train_rs = x_train.reshape(x_train.shape[0], -1)
    x_test_rs = x_test.reshape(x_test.shape[0], -1)
    clf = train_classifier(x_train_rs, y_train)
    report_model(clf, x_train_rs, x_test_rs, y_train, y_test)

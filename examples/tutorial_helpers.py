# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-examples file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from sklearn.metrics import roc_curve, auc
import matplotlib.pylab as plt
import ml_sampler
import pandas as pd
import numpy as np


def sample_stats(views, is_spam, biases, num_samples):
    """Output some statistics associated with taking a sample.
    Args:
        is_positive: numpy.array of values indicating if the record is a
            positive class.
        scores: numpy.array of values indicating the model score.
        num_samples: Number of samples to take.
        bias_func: Function to apply to @scores. Serves to bias the sampling
            procedure.

    Returns: a dictionary with the following entries
        'prevalence' - The estimate of the percentage of items that are
            positive in the overall population.
        'sampled_positive_percent' - The number of items sampled that are
            positive.
    """

    index, p_sample = ml_sampler.biased_sample(
        biases=biases,
        weights=views,
        num_samples=num_samples,
    )

    sample_weights = views[index]
    sample_is_spam = is_spam[index]

    est_pos_volume = ml_sampler.estimator(
        sample_weights,
        p_sample,
        sample_is_spam,
    )

    confidence_interval = ml_sampler.estimated_confidence_interval(
        sample_weights,
        p_sample,
        sample_is_spam,
    )

    # H-H Estimator of prevalence
    est_prevalence = est_pos_volume / views.sum() * 100.0
    confidence_interval = confidence_interval / views.sum() * 100.0

    true_prevalence = (views * is_spam).sum() / views.sum() * 100.0

    coverage = confidence_interval - true_prevalence


    # Percent of sampled entries that are positive. If this is greater
    #  than prevalence then we have over-sampled positive examples.
    sampled_positive_percent = sample_is_spam.mean() * 100.0

    return {
        'true_prevalence': true_prevalence,
        'est_prevalence': est_prevalence,
        'sampled_positive_percent': sampled_positive_percent,
        'lower_bound': confidence_interval[0],
        'upper_bound': confidence_interval[1],
        'bound_width': np.ptp(confidence_interval),
        'coverage': coverage[0] <= 0 and coverage[1] >= 0
    }


def simulated_sample_stats(views, is_spam, biases, num_samples,
                           num_iterations=4000):
    """Take a number of samples, analyze

    Args:
        is_positive: numpy.array of values indicating if the record is a
            positive class.
        scores: numpy.array of values indicating the model score.
        num_samples: Number of samples to take.

    Returns:
        DataFrame(true_prevalence, est_prevalence, positive_percent, ci_lower_bound, ci_upper_bound, bound_width, coverage)

    """
    result = []

    for _ in range(num_iterations):
        r = sample_stats(views, is_spam, biases, num_samples)
        result.append(r)

    return pd.DataFrame(result)


def get_auc(y_test, y_pred):
    """Get the ROC AUC for a set of true values and associated scores"""
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    return auc(fpr, tpr)


def plot_roc(y_test, y_pred, label=''):
    """Compute ROC curve and ROC area"""

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic' + label)
    plt.legend(loc="lower right")
    plt.show()


def print_confidence_interval(ci, tabs=''):
    """Pretty print confidence interval information"""
    ci = list(ci)
    ci += [np.ptp(ci)]

    print(tabs + 'Value: {1:.04f}'.format(*ci))
    print(tabs + '95% Confidence Interval: ({0:.04f}, {2:.04f})'.format(*ci))
    print(tabs + '\tCI Width: {3:.05f}'.format(*ci))

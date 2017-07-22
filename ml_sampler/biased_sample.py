# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def biased_sample(biases, weights, num_samples):
    """Take an weighted & based sample - then correct for the bias via
    weighting each sampled record.

    Args:
        biases: Value to bias the sample towards. Entries must be positive. If
            bias[0] == 2 and bias[1] == 1 the record at index 0 will be sampled
            2x as frequently as the record at index 1 but will have half the
            weight*. This is similar to the importance parameter in importance
            sampling.
        weights: np.array of record weights. This is the size (weight)
            in probability proportionate to size sampling. Entries must be
            positive.
        num_samples: Number of samples to take.

    Returns: A tuple of (sampled_index, weights, p_sample).
        sampled_ids: Indicates the entries that were sampled. np.array of length
        @num_samples.
        p_sample: Probability that each record was sampled.

    Notes:
        *If bias and weight arrays are correlated then the above frequencies
        and weights may not hold. The method will still produce an unbiased
        estimate.

        TODO: Come up with a more complete study of this. Provide a single step
        correction.
     """
    biases = np.array(biases)

    weights = np.array(weights)

    if np.sum(weights <= 0) > 0:
        raise ValueError('All weights must be strictly positive')

    if np.sum(biases <= 0) > 0:
        raise ValueError('All biases must be strictly positive')

    p_sample = biases * weights
    p_sample = p_sample / p_sample.sum()

    sampled_ids = np.random.choice(len(biases), min(num_samples, len(biases)),
                                   p=p_sample, replace=True)

    p_sample = p_sample

    return sampled_ids, p_sample[sampled_ids]


def estimator(weights, p_sample, is_positive):
    '''Hansen-Hurwitz Estimator
    Args:
        weights: numpy.array of weights assocated with the records.
            weights * is_positive are typcially refered to as y in the
            literature.
        p_sample: numpy.array of the probability the records were sampled.
            Typically refered to as pi in the literature.
        is_positve: numpy.array of boolean values indicating the positive
            records. weights * is_positive are typcially refered to as y in the
            literature.
    Returns:
        The extimated value for the larger population.
    '''
    n = weights.shape[0]
    return (weights / p_sample)[is_positive].sum() / n


def estimated_variance(weights, p_sample, is_positive):
    '''The estimated variance of the Hansen-Hurwitz estimator
    Args:
        weights: numpy.array of weights assocated with the records.
            weights * is_positive are typcially refered to as y in the
            literature.
        p_sample: numpy.array of the probability the records were sampled.
            Typically refered to as pi in the literature.
        is_positve: numpy.array of boolean values indicating the positive
            records. weights * is_positive are typcially refered to as y in the
            literature.
    Returns:
        The extimated variance for the larger population.
    '''
    n = weights.shape[0]
    T = estimator(weights, p_sample, is_positive)
    vals = (weights / p_sample * is_positive - T)**2
    return vals.sum() / n / (n - 1)


def estimated_confidence_interval(weights, p_sample, is_positive, z=1.96):
    var = estimated_variance(weights, p_sample, is_positive)
    stdev = np.sqrt(var)
    est = estimator(weights, p_sample, is_positive)
    return np.array([est - z * stdev, est + z*stdev])

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

    Returns: A tuple of (sampled_index, weights).
        sampled_ids: Indicates the entries that were sampled. np.array of length
        @num_samples.
        weights: Weight for each record specified in sampled_ids. This
            gives the Horvitz-Thompson Estimator for the sample. np.array of
            length @num_samples.

    Example:

        # Variables:
        #    weights - np.array specifying the size/weight of each record.
        #    is_positive - np.array specifying if the record is positive.
        #    biases - np.array specifying the bias for each record.

        sample_index, sample_weights = biased_sample(biases, 1000, weights)

        # the estimate of the volume of 'positive' things in the population
        sample_weights[is_positive[sample_index]].sum()

        # the estimate of the percent of 'positive' things in the population
        sample_weights[is_positive[sample_index]].sum() / weights.sum()

        # the percent of positive examples sampled
        is_positive[sample_index].mean() * 100.0

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
        raise ValueError('All weights must be positive')

    if np.sum(biases <= 0) > 0:
        raise ValueError('All biases must be positive')

    p_sample = biases * weights
    p_sample = p_sample / p_sample.sum()

    sampled_ids = np.random.choice(len(biases), min(num_samples, len(biases)),
                                   p=p_sample, replace=True)

    # This gives the Horvitz-Thompson Estimator
    # sum up weights to get to the percent of prevalence
    #  weights[is_interesting].sum() == the volume of interesting things
    #  in other words this is the estimate of weight[is_interesting].sum()
    sampled_weights = (weights[sampled_ids] / p_sample[sampled_ids])
    sampled_weights = sampled_weights / num_samples

    return sampled_ids, sampled_weights

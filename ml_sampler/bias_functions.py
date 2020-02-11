# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""Methods for introducing bias into a sample"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import range
import numpy as np
import scipy.interpolate

def _merge_infrequent_bins(scores, bins, tolerance):
    """Adjust scores so that no bin has less than a fixed percentage of
    records. interpolated_pdf_reciprocal and histogram_reciprocal methods
    are sensitive to low frequency bins. This method identifies bins that
    have low counts and adjusts the scores such that the score will
    now show up in the nearest adjacent bin that is above the threshold.

    Args:
        scores: np.array of scores to bias the sample with
        bins: np.array of values signifying the binning for @scores
        tolerance: value specifying the percentage threshold at which
            to start merging bins. 0.005 means merge bins that have
            less than 0.05% of the population.

    Returns: np.array of bins
    """

    scores = np.array(scores)

    num_bins = len(bins) - 1

    counts, bins = np.histogram(scores, bins=bins)
    counts = counts * 1.0 / counts.sum()

    posts_to_remove = []
    last_merged = False
    for ind in range(num_bins - 1):
        if not last_merged and counts[ind] < tolerance:
            posts_to_remove.append(ind + 1)
            last_merged = True
        else:
            last_merged = False

    posts_to_keep = set(range(num_bins + 1)).difference(posts_to_remove)
    posts_to_keep = list(posts_to_keep)

    if bins.size != bins[posts_to_keep].size:
        return _merge_infrequent_bins(scores, bins[posts_to_keep], tolerance)

    return bins[posts_to_keep]


def interpolated_pdf_reciprocal(
    scores, bins=None, merge_threshold=0.01, histogram_weights=None
):
    """Attempts to take equal samples from each bin. It does this by
    constructing an (interpolated) PDF from @scores and then returning
    1 / pdf(score).

    Args:
        scores: np.array of scores to bias the sample with
        bins: np.array of values signifying the binning for @scores.
            Defaults to 10 bins linearly spaced over scores.
        merge_threshold: merge bins that have less than some fraction
            of the overall population. Low population bins can significantly
            hurt performance of this method. Defaults to 0.01 which
            represents 1.0% of the population. Make this False to turn off.
            TODO: make merge_threshold d
        histogram_weights: np.array of weights. Each value only contributes its
            associated weight towards the bin counts in the histogram. The
            default weights are np.ones()

    Returns: np.array of bias weights for each record
    """

    if bins is None:
        # reasonable default
        bins = np.linspace(scores.min(), scores.max(), 10)

    if histogram_weights is None:
        # set to unit histogram weights
        histogram_weights = np.ones(len(scores))

    if merge_threshold:
        bins = _merge_infrequent_bins(scores, bins, merge_threshold)

    counts, bins = np.histogram(scores, bins=bins, weights=histogram_weights)

    bin_posts = [(bins[i] + bins[i - 1]) / 2.0 for i in range(1, len(bins))]

    if len(bin_posts) <= 1:
        return np.ones(len(scores))

    f = scipy.interpolate.interp1d(bin_posts, counts, bounds_error=False,
                                   fill_value=(counts[0], counts[-1]))

    pdf_values = 1.0 / f(scores)

    return pdf_values


def histogram_reciprocal(
    scores, bins=None, merge_threshold=0.01, histogram_weights=None
):
    """Attempts to take equal samples from each bin. It does this by
    constructing a histogram from @scores and then returning
    1 / histogram_value(score).

    Args:
        scores: np.array of scores to bias the sample with
        bins: np.array of values signifying the binning for @scores.
            Defaults to 10 bins linearly spaced over scores.
        merge_threshold: merge bins that have less than some fraction
            of the overall population. Low population bins can significantly
            hurt performance of this method. Defaults to 0.01 which
            represents 1.0% of the population. Make this False to turn off.
        histogram_weights: np.array of weights. Each value only contributes its
            associated weight towards the bin counts in the histogram. The
            default weights are np.ones()

    Returns: np.array of bias weights for each record
    """

    if bins is None:
        # reasonable default
        bins = np.linspace(scores.min(), scores.max(), 10)

    if histogram_weights is None:
        # set to unit histogram weights
        histogram_weights = np.ones(len(scores))

    if merge_threshold:
        bins = _merge_infrequent_bins(scores, bins, merge_threshold)

    counts, bins = np.histogram(scores, bins=bins)

    if len(bins) <= 1:
        return np.ones(len(scores))

    index = np.searchsorted(bins, scores, side='left') - 1

    pdf_values = 1.0 / counts[index]

    return pdf_values


def bin_weights_raw(scores, bins, bin_weights):
    """Assign raw weights to each bin. Does not correct for the raw
    frequency of items found in each bin.

    Args:
        scores: np.array of scores to bias the sample.
        bins: np.array of values signifying the binning for @scores.

    Returns: np.array of bias weights for each record
    """
    index = np.searchsorted(bins, scores, side='left') - 1
    return bin_weights[index] / bin_weights.max()


def bin_weights_corrected(scores, bins, bin_weights):
    """Applies weights to each bin but first corrects for the frequency of
    items within a bin. This has the effect of attempting to sample
    proportionate to the values specified in bin_weights.

    Args:
        scores: np.array of scores to bias the sample.
        bins: np.array of values signifying the binning for @scores.

    Returns: np.array of bias weights for each record.
    """

    return bin_weights_raw(scores, bins, bin_weights) * \
            histogram_reciprocal(scores, bins)

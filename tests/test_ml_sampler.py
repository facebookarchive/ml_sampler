# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Test ml sampler"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest
import ml_sampler
import numpy as np


class MLSamplerTest(unittest.TestCase):
    """End to end tests for ml_sampler"""

    def equal_assert(self, est_prevalence):
        self.assertAlmostEqual(self.prevalence / est_prevalence, 1.0, delta=0.08)

    def setUp(self):
        np.random.seed(1)

        size = 1000000
        importance = np.random.exponential(scale=100, size=size)
        is_positive = np.random.random(size) < 0.2

        scores = np.zeros(size)
        scores[is_positive] = np.random.normal(loc=1.0, scale=1.0,
                                               size=is_positive.sum())
        scores[~is_positive] = np.random.normal(loc=0.8, scale=1.0,
                                                size=(~is_positive).sum())
        scores += scores.min()

        self.size = size
        self.importance = importance
        self.is_positive = is_positive
        self.scores = scores

        self.prevalence = importance[is_positive].sum() / importance.sum()

        self.num_samples = 10000

    def test_sample(self):

        sample_index, sample_weights = ml_sampler.biased_sample(
            biases=np.ones(self.size),
            weights=self.importance,
            num_samples=self.num_samples
        )

        est_prevalence = sample_weights[self.is_positive[sample_index]].sum()
        est_prevalence /= self.importance.sum()
        self.equal_assert(est_prevalence)

    def test_pdf(self):

        sample_index, sample_weights = ml_sampler.biased_sample(
            biases=ml_sampler.interpolated_pdf_reciprocal(self.scores),
            weights=self.importance,
            num_samples=self.num_samples
        )

        est_prevalence = sample_weights[self.is_positive[sample_index]].sum()
        est_prevalence /= self.importance.sum()
        self.equal_assert(est_prevalence)

    def test_hist(self):

        sample_index, sample_weights = ml_sampler.biased_sample(
            biases=ml_sampler.histogram_reciprocal(self.scores),
            weights=self.importance,
            num_samples=self.num_samples
        )

        est_prevalence = sample_weights[self.is_positive[sample_index]].sum()
        est_prevalence /= self.importance.sum()
        self.equal_assert(est_prevalence)

    def test_bin_weights(self):

        bins = np.linspace(self.scores.min(), self.scores.max(), 10)

        bias = ml_sampler.bin_weights_corrected(
            self.scores,
            bins,
            bin_weights=np.linspace(1, 10, 10)
        )

        sample_index, sample_weights = ml_sampler.biased_sample(
            biases=bias,
            weights=self.importance,
            num_samples=self.num_samples
        )

        est_prevalence = sample_weights[self.is_positive[sample_index]].sum()
        est_prevalence /= self.importance.sum()
        self.equal_assert(est_prevalence)

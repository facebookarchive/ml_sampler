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

from builtins import str
from biased_sample import biased_sample
from bias_functions import interpolated_pdf_reciprocal

import numpy as np
import pandas as pd

import argparse


def main():
    """
    input file format: id, volume, weight, uniform_score, histogram_weights (these columns must
    appear in that order as the first four columns, followed by any additional
    columns)
    (tabs not commas)
        id - the unique id of the record
        volume - the volume that the record represents. For ads this could
            be the number of impressions the ad got. If we were trying to
            measure the percent of impressions.
        weight - weight we want to oversample this record by. Later we will
            downweight this sample so that the math works out right.
        uniform_score - score to sample uniformly across.
        histogram_weights - weights for np.histogram in interpolated_pdf_reciprocal.
            Each value only contriubtes it's associated weight to the bins

    output file format: id, est_volume, y, pi (followed by any additional
    columns that appeared in the input file)
    (tabs not commas)

        y - the 'y' perameter in the Horvitz-Thompson equations.
            The weight of the record.
        pi - the 'pi' perameter in the Horvitz-Thompson equations.
            The probability that the record was sampled.
        est_volume - represents the volume that id represents. This
            is actually for convienence. It is equivalent to y/pi.
        weight - the original input weight
        uniform_score - the original input uniform_score

    sum(est_volume) == sum(sum_volume) # approximately
    """

    args = get_args()

    num_samples = int(args.num_samples)
    num_bins = int(args.num_bins) + 1
    merge_threshold = float(args.merge_threshold)
    file_name = args.file_name

    assert num_samples > 0, 'num_samples must be a positive integer'
    assert num_bins > 0, 'num_bins must be a positive integer'
    assert merge_threshold > 0, 'merge_threshold must be a positive number'

    df_all = pd.read_csv(
        file_name,
        sep='\t',
    )

    columns = ['ids', 'volumes', 'weights', 'uniform_scores', 'histogram_weights']
    df = df_all.iloc[:, :len(columns)]
    df.columns = columns

    if any(df.duplicated('ids')):
        raise ValueError('Input must contain unique ids')

    df_rest = df_all.iloc[:, len(columns):]

    ids = df.ids.values
    volumes = df.volumes.values
    weights = df.weights.values
    uniform_scores = df.uniform_scores.values
    histogram_weights = df.histogram_weights.values

    other_input_values = df_rest.as_matrix()

    if len(ids) == 0:
        raise ValueError('No non-NULL lines found in stdin.')

    bins = np.linspace(uniform_scores.min(), uniform_scores.max(), num_bins)
    biases = interpolated_pdf_reciprocal(
        uniform_scores,
        bins=bins,
        merge_threshold=merge_threshold,
        histogram_weights=histogram_weights
    )
    biases *= weights
    sample_index, p_sample = biased_sample(
        biases=biases,
        # the thing we call volume here, is used as a weight in the sample
        # this is to be consistent with stats literature
        weights=volumes,
        num_samples=num_samples,
        with_replacement=args.with_replacement
    )

    sample_volumes = volumes[sample_index]
    sample_weights = weights[sample_index]
    sample_uniform_scores = uniform_scores[sample_index]

    est_volume = sample_volumes / p_sample / num_samples

    rows = []
    for i, index in enumerate(sample_index):
        sample_id = ids[index]
        row = []
        row.append(sample_id)
        row.append(est_volume[i])
        row.append(sample_volumes[i])
        row.append(p_sample[i])
        row.append(sample_weights[i])
        row.append(sample_uniform_scores[i])
        extra_input_data = other_input_values[i]

        for input_val in extra_input_data:
            row.append(input_val)

        row = [str(x) for x in row]
        rows.append('\t'.join(row))

    print('\n'.join(rows))


def get_args():
    """ Get command line args. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'num_samples',
        help='The number of samples in the final output.',
    )
    parser.add_argument(
        'num_bins',
        help='The number of bins to sample from.',
    )
    parser.add_argument(
        'merge_threshold',
        help='Minimum threshold for merging bings.',
    )
    parser.add_argument(
        'file_name',
        help='A tsv file containing records to be sampled.'
    )
    parser.add_argument(
        'with_replacement',
        help="""Flag to determine if the sampling is with
                replacement (True, default) or not (False)""",
        default=True,
        type=bool,
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()

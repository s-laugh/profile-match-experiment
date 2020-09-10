import io
import numpy as np
import pandas as pd
import itertools

# import modules necessary for schema definition
import clkhash
from clkhash.field_formats import *
from clkhash.schema import Schema
from clkhash.comparators import NgramComparison

from anonlinkclient.utils import generate_clk_from_csv, generate_candidate_blocks_from_csv, combine_clks_blocks

import recordlinkage
from recordlinkage.datasets import load_febrl4

dfA, dfB = load_febrl4()
print(dfA.head())
print(dfA.columns)
a_csv = io.StringIO()
dfA.to_csv(a_csv)

""" fields = [
    Ignore('rec_id'),
    StringSpec('given_name', FieldHashingProperties(comparator=NgramComparison(2), strategy=BitsPerFeatureStrategy(300))),
    StringSpec('surname', FieldHashingProperties(comparator=NgramComparison(2), strategy=BitsPerFeatureStrategy(300))),
    IntegerSpec('street_number', FieldHashingProperties(comparator=NgramComparison(1, True), strategy=BitsPerFeatureStrategy(300), missing_value=MissingValueSpec(sentinel=''))),
    StringSpec('address_1', FieldHashingProperties(comparator=NgramComparison(2), strategy=BitsPerFeatureStrategy(300))),
    StringSpec('address_2', FieldHashingProperties(comparator=NgramComparison(2), strategy=BitsPerFeatureStrategy(300))),
    StringSpec('suburb', FieldHashingProperties(comparator=NgramComparison(2), strategy=BitsPerFeatureStrategy(300))),
    IntegerSpec('postcode', FieldHashingProperties(comparator=NgramComparison(1, True), strategy=BitsPerFeatureStrategy(300))),
    StringSpec('state', FieldHashingProperties(comparator=NgramComparison(2), strategy=BitsPerFeatureStrategy(300))),
    IntegerSpec('date_of_birth', FieldHashingProperties(comparator=NgramComparison(1, True), strategy=BitsPerFeatureStrategy(300), missing_value=MissingValueSpec(sentinel=''))),
    Ignore('soc_sec_id')
] """

fields = [
    Ignore('rec_id'),
    StringSpec('given_name', FieldHashingProperties(comparator=NgramComparison(2), strategy=BitsPerFeatureStrategy(200))),
    StringSpec('surname', FieldHashingProperties(comparator=NgramComparison(2), strategy=BitsPerFeatureStrategy(200))),
    IntegerSpec('street_number', FieldHashingProperties(comparator=NgramComparison(1, True), strategy=BitsPerFeatureStrategy(100), missing_value=MissingValueSpec(sentinel=''))),
    StringSpec('address_1', FieldHashingProperties(comparator=NgramComparison(2), strategy=BitsPerFeatureStrategy(100))),
    StringSpec('address_2', FieldHashingProperties(comparator=NgramComparison(2), strategy=BitsPerFeatureStrategy(100))),
    StringSpec('suburb', FieldHashingProperties(comparator=NgramComparison(2), strategy=BitsPerFeatureStrategy(100))),
    IntegerSpec('postcode', FieldHashingProperties(comparator=NgramComparison(1, True), strategy=BitsPerFeatureStrategy(100))),
    StringSpec('state', FieldHashingProperties(comparator=NgramComparison(2), strategy=BitsPerFeatureStrategy(100))),
    IntegerSpec('date_of_birth', FieldHashingProperties(comparator=NgramComparison(1, True), strategy=BitsPerFeatureStrategy(200), missing_value=MissingValueSpec(sentinel=''))),
    Ignore('soc_sec_id')
]

secret = 'secret'
schema = Schema(fields, 1024)

a_csv.seek(0)
clks_a = generate_clk_from_csv(a_csv, secret, schema)

print(len(clks_a))

b_csv = io.StringIO()
dfB.to_csv(b_csv)
b_csv.seek(0)
clks_b = generate_clk_from_csv(b_csv, secret, schema)

print(len(clks_b))

# find matches

from anonlinkclient.utils import deserialize_filters
from bitarray import bitarray
import base64

import anonlink

def mapping_from_clks(clks_a, clks_b, threshold):
    results_candidate_pairs = anonlink.candidate_generation.find_candidate_pairs(
            [clks_a, clks_b],
            anonlink.similarities.dice_coefficient,
            threshold
    )
    solution = anonlink.solving.greedy_solve(results_candidate_pairs)
    print('Found {} matches'.format(len(solution)))
    # each entry in `solution` looks like this: '((0, 4039), (1, 2689))'.
    # The format is ((dataset_id, row_id), (dataset_id, row_id))
    # As we only have two parties in this example, we can remove the dataset_ids.
    # Also, turning the solution into a set will make it easier to assess the
    # quality of the matching.
    return set((a, b) for ((_, a), (_, b)) in solution)

found_matches = mapping_from_clks(clks_a, clks_b, 0.9)

# rec_id in dfA has the form 'rec-1070-org'. We only want the number. Additionally, as we are
# interested in the position of the records, we create a new index which contains the row numbers.
dfA_ = dfA.rename(lambda x: x[4:-4], axis='index').reset_index()
dfB_ = dfB.rename(lambda x: x[4:-6], axis='index').reset_index()
# now we can merge dfA_ and dfB_ on the record_id.
a = pd.DataFrame({'ida': dfA_.index, 'rec_id': dfA_['rec_id']})
b = pd.DataFrame({'idb': dfB_.index, 'rec_id': dfB_['rec_id']})
dfj = a.merge(b, on='rec_id', how='inner').drop(columns=['rec_id'])
# and build a set of the corresponding row numbers.
true_matches = set((row[0], row[1]) for row in dfj.itertuples(index=False))

def describe_matching_quality(found_matches, show_examples=False):
    if show_examples:
        print('idx_a, idx_b,     rec_id_a,       rec_id_b')
        print('---------------------------------------------')
        for a_i, b_i in itertools.islice(found_matches, 10):
            print('{:4d}, {:5d}, {:>11}, {:>14}'.format(a_i+1, b_i+1, a.iloc[a_i]['rec_id'], b.iloc[b_i]['rec_id']))
        print('---------------------------------------------')

    tp = len(found_matches & true_matches)
    fp = len(found_matches - true_matches)
    fn = len(true_matches - found_matches)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print('Precision: {:.3f}, Recall: {:.3f}'.format(precision, recall))

describe_matching_quality(found_matches, show_examples=True)

found_matches = mapping_from_clks(clks_a, clks_b, 0.8)
describe_matching_quality(found_matches, show_examples=True)

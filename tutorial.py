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

fields = [
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
]

schema = Schema(fields, 1024)

secret = 'secret'

# NBVAL_IGNORE_OUTPUT
a_csv.seek(0)
clks_a = generate_clk_from_csv(a_csv, secret, schema)

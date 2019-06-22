import collections
import logging
import mmh3
import re
import resource

import numpy as np
import pandas as pd
import six
import tqdm
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer

from citeomatic.candidate_selectors import CandidateSelector
from citeomatic.utils import flatten
from citeomatic.common import DatasetPaths
from citeomatic.models.options import ModelOptions

dp = DatasetPaths()

CLEAN_TEXT_RE = re.compile('[^ a-z]')

# filters for authors and docs
MAX_AUTHORS_PER_DOCUMENT = 8
MAX_KEYPHRASES_PER_DOCUMENT = 20
MIN_TRUE_CITATIONS = {
    'pubmed': 2,
    'dblp': 1,
    'oc': 2
}
MAX_TRUE_CITATIONS = 100

# Adjustments to how we boost heavily cited documents.
CITATION_SLOPE = 0.01
MAX_CITATION_BOOST = 0.02

# Parameters for soft-margin data generation.
TRUE_CITATION_OFFSET = 0.3
HARD_NEGATIVE_OFFSET = 0.2
NN_NEGATIVE_OFFSET = 0.1
EASY_NEGATIVE_OFFSET = 0.0

# ANN jaccard percentile cutoff
ANN_JACCARD_PERCENTILE = 0.05


def label_for_doc(d, offset):
    sigmoid = 1 / (1 + np.exp(-d.in_citation_count * CITATION_SLOPE))
    return offset + (sigmoid * MAX_CITATION_BOOST)


def jaccard(featurizer, x, y):
    x_title, x_abstract = featurizer._cleaned_document_words(x)
    y_title, y_abstract = featurizer._cleaned_document_words(y)
    a = set(x_title + x_abstract)
    b = set(y_title + y_abstract)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def _clean(text):
    return CLEAN_TEXT_RE.sub(' ', text.lower())




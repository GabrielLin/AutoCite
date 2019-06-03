from allennlp.data.fields import Field, TextField
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.vocabulary import Vocabulary

from torch import tensor

import numpy as np

from typing import Dict, List, Iterator

from citeomatic.neighbors import ANN

import re
from itertools import compress

CLEAN_TEXT_RE = re.compile('[^ a-z]')

class ScalarField(Field):
    def __init__(self, scalar: int) -> None:
        super().__init__()
        self.scalar = scalar

    def as_tensor(self, padding_lengths: Dict[str, int]) -> tensor:
        return tensor([self.scalar])

    def empty_field(self):
        return tensor([])

    def get_padding_lengths(self):
        return {}

# filters for authors and docs
MAX_AUTHORS_PER_DOCUMENT = 8
MAX_KEYPHRASES_PER_DOCUMENT = 20

MAX_TRUE_CITATIONS = 100
MIN_TRUE_CITATIONS = 2

# Adjustments to how we boost heavily cited documents.
CITATION_SLOPE = 0.01
MAX_CITATION_BOOST = 0.02

# Parameters for soft-margin data generation.
TRUE_CITATION_OFFSET = 0.3
HARD_NEGATIVE_OFFSET = 0.2
NN_NEGATIVE_OFFSET = 0.1
EASY_NEGATIVE_OFFSET = 0.0

ANN_JACCARD_PERCENTILE = 0.05

NEG_TO_POS_RATIO = 6

STOPWORDS = {
'abstract', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the',
'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with',
'the', 'we', 'our', 'which'
}

KEYS = ['hard_negatives', 'nn', 'easy']

margin_multiplier = 1

margins_offset_dict = {
    'true': TRUE_CITATION_OFFSET * margin_multiplier,
    'hard': HARD_NEGATIVE_OFFSET * margin_multiplier,
    'nn': NN_NEGATIVE_OFFSET * margin_multiplier,
    'easy': EASY_NEGATIVE_OFFSET * margin_multiplier
}

column_index = {"index":0,"id":1,"inCitations":2,"outCitations":3,"abstract":4,"title":5}

class SimpleReader(DatasetReader):    
    def __init__(self, corpus, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=True)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.re = re.compile('[^ a-z]')
        self.corpus = corpus
        
    def text_to_instance(self, query_title: List[Token],
                         query_abstract: List[Token],
                         query_id: int = None,
                         candidate_title: List[Token] = None,
                         candidate_abstract: List[Token] = None,
                         label: float = None,
                         candidate_citations: float = None,
                         title_intersection: float = None,
                         abstract_intersection: float = None,
                         similarity: float = None) -> Instance:
        
        query_title_field = TextField(query_title, self.token_indexers)
        query_abstract_field = TextField(query_abstract, self.token_indexers)

        fields = {"query_title": query_title_field, "query_abstract": query_abstract_field}

        has_candidate = candidate_title is not None and candidate_abstract is not None

        if has_candidate:
            candidate_title_field = TextField(candidate_title, self.token_indexers)
            candidate_abstract_field = TextField(candidate_abstract, self.token_indexers)

            fields["candidate_title"] = candidate_title_field
            fields["candidate_abstract"] = candidate_abstract_field

        if label is not None:
            fields["label"] = ScalarField(label)

        has_nnrank_features = (candidate_citations is not None
                               and title_intersection is not None
                               and abstract_intersection is not None
                               and similarity is not None)

        if has_nnrank_features:
            fields["candidate_citations"] = ScalarField(candidate_citations)
            fields["title_intersection"] = ScalarField(title_intersection)
            fields["abstract_intersection"] = ScalarField(abstract_intersection)
            fields["cos_sim"] = ScalarField(similarity)
        elif query_id is not None:
            query_id_field = ScalarField(query_id)
            fields["query_id"] = query_id_field

        return Instance(fields)
    
    def _read(self,str="") -> Iterator[Instance]:
        for row in self.corpus.itertuples():
            query_title = self._tokenize_text(row[column_index["title"]])
            query_abstract = self._tokenize_text(row[column_index["abstract"]])
            query_id = row[column_index["index"]]
            
            yield self.text_to_instance(query_title, query_abstract, query_id)
            
    def _tokenize_text(self, text: str) -> List[Token]:
        tokens = CLEAN_TEXT_RE.sub(' ', text.lower()).split()
        mask = [word not in STOPWORDS for word in tokens]
        return [Token(word) for word in compress(tokens,mask)]

    def _jaccard(self, x_title, x_abstract, y_title, y_abstract) -> float:
        x_title = [str(t) for t in x_title]
        x_abstract = [str(t) for t in x_abstract]
        y_title = [str(t) for t in y_title]
        y_abstract = [str(t) for t in y_abstract]

        a = set(x_title + x_abstract)
        b = set(y_title + y_abstract)
        c = a.intersection(b)
        if len(a)+len(b) == len(c):
            return 0
        else:
            return float(len(c)) / (len(a) + len(b) - len(c))

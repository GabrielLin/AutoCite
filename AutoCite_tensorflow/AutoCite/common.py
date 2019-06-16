import importlib
import os

import pickle

from citeomatic import file_util
from citeomatic.schema_pb2 import Document as ProtoDoc
import spacy
from whoosh.fields import *

PAPER_EMBEDDING_MODEL = 'paper_embedder'
CITATION_RANKER_MODEL = 'citation_ranker'

nlp = spacy.load("en")
RESTRICTED_POS_TAGS = {'PUNCT', 'SYM', 'DET', 'NUM', 'SPACE', 'PART'}

schema = Schema(title=TEXT,
                abstract=TEXT,
                id=ID(stored=True))


def global_tokenizer(text, restrict_by_pos=False, lowercase=True, filter_empty_token=True):
    if restrict_by_pos:
        token_list = [
            w.text for w in nlp(text) if w.pos_ not in RESTRICTED_POS_TAGS
        ]
    else:
        token_list = [w.text for w in nlp(text)]

    if lowercase:
        token_list = [w.lower() for w in token_list]

    if filter_empty_token:
        token_list = [w for w in token_list if len(w) > 0]

    return token_list



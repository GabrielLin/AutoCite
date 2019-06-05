import importlib
import os

import pickle

#from citeomatic import file_util
from citeomatic.schema_pb2 import Document as ProtoDoc
#import spacy
#from whoosh.fields import *

PAPER_EMBEDDING_MODEL = 'paper_embedder'
CITATION_RANKER_MODEL = 'citation_ranker'

#nlp = spacy.load("en")
#RESTRICTED_POS_TAGS = {'PUNCT', 'SYM', 'DET', 'NUM', 'SPACE', 'PART'}

'''schema = Schema(title=TEXT,
                abstract=TEXT,
                id=ID(stored=True))'''


'''def global_tokenizer(text, restrict_by_pos=False, lowercase=True, filter_empty_token=True):
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

    return token_list'''


class FieldNames(object):
    PAPER_ID = "id"
    TITLE = "title"
    ABSTRACT = "paperAbstract"
    AUTHORS = "authors"

    VENUE = "venue"
    YEAR = "year"

    IN_CITATIONS = "inCitations"
    OUT_CITATIONS = "outCitations"
    KEY_PHRASES = "entities"

    URLS = "s2PdfUrl"
    S2_URL = "s2Url"

    #OUT_CITATION_COUNT = 'out_citation_count'
    #IN_CITATION_COUNT = 'in_citation_count'

    #DATE = 'date'

    TITLE_RAW = "title"
    ABSTRACT_RAW = "paperAbstract"


class DatasetPaths(object):
    BASE_DIR = os.path.abspath("./data")

    DBLP_GOLD_DIR = os.path.join(BASE_DIR, 'comparison/dblp/gold')
    DBLP_CORPUS_JSON = os.path.join(BASE_DIR, 'comparison/dblp/corpus.json')
    DBLP_DB_FILE = os.path.join(BASE_DIR, 'db/dblp.sqlite.db')
    DBLP_BM25_INDEX = os.path.join(BASE_DIR, 'bm25_index/dblp/')

    PUBMED_GOLD_DIR = os.path.join(BASE_DIR, 'comparison/pubmed/gold')
    PUBMED_CORPUS_JSON = os.path.join(BASE_DIR, 'comparison/pubmed/corpus.json')
    PUBMED_DB_FILE = os.path.join(BASE_DIR, 'db/pubmed.sqlite.db')
    PUBMED_BM25_INDEX = os.path.join(BASE_DIR, 'bm25_index/pubmed/')

    OC_FILE = os.path.join(BASE_DIR, 'open_corpus/papers-2017-02-21.json.gz')
    OC_CORPUS_JSON = os.path.join(BASE_DIR, 'open_corpus/corpus.json')
    OC_DB_FILE = os.path.join(BASE_DIR, 'db/oc.sqlite.db')
    OC_BM25_INDEX = os.path.join(BASE_DIR, 'bm25_index/oc/')
    OC_PKL_FILE = os.path.join(BASE_DIR, 'open_corpus/corpus.pkl')
    OC_ANN_FILE = os.path.join(BASE_DIR, 'open_corpus/ann.pkl')

    PRETRAINED_DIR = os.path.join(BASE_DIR, 'pretrained')
    EMBEDDING_WEIGHTS_FILENAME = 'embedding.h5'
    PRETRAINED_VOCAB_FILENAME = 'vocab.txt'
    FEATURIZER_FILENAME = 'featurizer.pickle'
    OPTIONS_FILENAME = 'options.json'
    CITEOMATIC_WEIGHTS_FILENAME = 'weights.h5'

    def embeddings_weights_for_corpus(self, corpus_name):
        return os.path.join(
            self.PRETRAINED_DIR,
            corpus_name + '_' + self.EMBEDDING_WEIGHTS_FILENAME
        )

    def vocab_for_corpus(self, corpus_name):
        return os.path.join(
            self.PRETRAINED_DIR,
            corpus_name + '_' + self.PRETRAINED_VOCAB_FILENAME
        )

    def get_json_path(self, corpus_name):
        if corpus_name.lower() == 'dblp':
            return self.DBLP_CORPUS_JSON
        elif corpus_name.lower() == 'pubmed':
            return self.PUBMED_CORPUS_JSON
        elif (corpus_name.lower() == 'oc'
              or corpus_name.lower() == 'open_corpus'
              or corpus_name.lower() == 'opencorpus'):
            return self.OC_CORPUS_JSON
        else:
            return None

    def get_bm25_index_path(self, corpus_name):
        if corpus_name.lower() == 'dblp':
            return self.DBLP_BM25_INDEX
        elif corpus_name.lower() == 'pubmed':
            return self.PUBMED_BM25_INDEX
        elif (corpus_name.lower() == 'oc'
              or corpus_name.lower() == 'open_corpus'
              or corpus_name.lower() == 'opencorpus'):
            return self.OC_BM25_INDEX
        else:
            return None

    def get_db_path(self, corpus_name):
        if corpus_name.lower() == 'dblp':
            return self.DBLP_DB_FILE
        elif corpus_name.lower() == 'pubmed':
            return self.PUBMED_DB_FILE
        elif (corpus_name.lower() == 'oc'
              or corpus_name.lower() == 'open_corpus'
              or corpus_name.lower() == 'opencorpus'):
            return self.OC_DB_FILE
        else:
            return None

    def get_pkl_path(self, corpus_name):
        if (corpus_name.lower() == 'oc'
            or corpus_name.lower() == 'open_corpus'
            or corpus_name.lower() == 'opencorpus'):
            return self.OC_PKL_FILE
        else:
            assert False

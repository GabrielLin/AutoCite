import json
import logging
import sqlite3

import pickle
import tqdm

from citeomatic import file_util
from citeomatic.common import FieldNames, Document, DatasetPaths
from citeomatic.utils import batchify
from citeomatic.schema_pb2 import Document as ProtoDoc



class Corpus(object):

    training_ranges = {
        'dblp': (1966, 2007),  # both years inclusive
        'pubmed': (1966, 2008),  # both years inclusive
        'oc': None
    }
    validation_ranges = {'dblp': (2008, 2008), 'pubmed': (2009, 2009), 'oc': None}
    testing_ranges = {'dblp': (2009, 2011), 'pubmed': (2010, 2013), 'oc': None}

    def _fetch_paper_ids(self, date_range=None):
        """
        Fetch paper ids from sqlite db published between the provided years. If none provided,
        get all IDs
        :param date_range: a tuple of start and end year (both inclusive)
        :return:
        """
        if date_range is None:
            query = '''
                SELECT key from ids
                ORDER BY year
            '''
        else:
            query = '''
            SELECT key from ids 
            WHERE year >= {} AND year <= {} 
            ORDER BY year
            '''.format(date_range[0], date_range[1])

        id_rows = self._conn.execute(
            query
        ).fetchall()

        all_ids = [r[0] for r in id_rows]
        return all_ids

    def __init__(self, data_path, train_frac):
        self._conn = sqlite3.connect(
            data_path, check_same_thread=False, uri=True
        )
        self.train_frac = train_frac

        if 'dblp' in data_path:
            self.corpus_type = 'dblp'
        elif 'pubmed' in data_path:
            self.corpus_type = 'pubmed'
        else:
            self.corpus_type = 'oc'

        if self.corpus_type is 'oc':
            all_ids = self._fetch_paper_ids()
            n = len(all_ids)
            n_train = int(self.train_frac * n)
            n_valid = (n - n_train) // 2
            n_test = n - n_train - n_valid
            self.train_ids = all_ids[0:n_train]
            self.valid_ids = all_ids[n_train:n_train + n_valid]
            self.test_ids = all_ids[n_train + n_valid:]
        else:
            self.train_ids = self._fetch_paper_ids(Corpus.training_ranges[self.corpus_type])
            self.valid_ids = self._fetch_paper_ids(Corpus.validation_ranges[self.corpus_type])
            self.test_ids = self._fetch_paper_ids(Corpus.testing_ranges[self.corpus_type])

            n_train = len(self.train_ids)
            n_valid = len(self.valid_ids)
            n_test = len(self.test_ids)

        self.n_docs = len(self.train_ids) + len(self.valid_ids) + len(self.test_ids)
        self.all_ids = self.train_ids + self.valid_ids + self.test_ids
        self._id_set = set(self.all_ids)

        logging.info('%d training docs' % n_train)
        logging.info('%d validation docs' % n_valid)
        logging.info('%d testing docs' % n_test)

        logging.info("Loading documents into memory")
        self.documents = [doc for doc in self._doc_generator()]
        self.doc_id_to_index_dict = {doc.id: idx for idx, doc in enumerate(self.documents)}

    @staticmethod
    def load(data_path, train_frac=0.80):
        return load(data_path, train_frac)

    @staticmethod
    def load_pkl(corpus_pkl_location):
        return pickle.load(open(corpus_pkl_location, "rb"))

    @staticmethod
    def build(db_filename, source_json):
        return build_corpus(db_filename, source_json)

    def _doc_generator(self):
        with self._conn as tx:
            for row in tx.execute(
                    'SELECT payload from documents ORDER BY year'
            ):
                doc = ProtoDoc()
                doc.ParseFromString(row[0])
                yield Document.from_proto_doc(doc)

    def __len__(self):
        return self.n_docs

    def __iter__(self):
        for doc in self.documents:
            yield doc

    def __contains__(self, id):
        return id in self._id_set

    def __getitem__(self, id):
        index = self.doc_id_to_index_dict[id]
        return self.documents[index]

    def select(self, id_set):
        for doc in self.documents:
            if doc in id_set:
                yield doc.id, doc

    def filter(self, id_set):
        return self._id_set.intersection(id_set)

    def get_citations(self, doc_id):
        out_citations = self[doc_id].out_citations
        # Remove cited documents that appear after the year of publication of source document as
        # they indicate incorrect data
        return [cit_doc_id for cit_doc_id in out_citations if cit_doc_id in self._id_set and
                self[cit_doc_id].year <= self[doc_id].year]

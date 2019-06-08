import json
import logging
import sqlite3

import pickle
import tqdm

from citeomatic import file_util
from citeomatic.common import FieldNames, Document, DatasetPaths
from citeomatic.utils import batchify
from citeomatic.schema_pb2 import Document as ProtoDoc


def stream_papers(data_path):
    for line_json in tqdm.tqdm(file_util.read_json_lines(data_path)):
        citations = set(line_json[FieldNames.OUT_CITATIONS])
        citations.discard(line_json[FieldNames.PAPER_ID])  # remove self-citations
        citations = list(citations)

        in_citation_count = len(line_json[FieldNames.IN_CITATIONS])

        key_phrases = list(set(line_json[FieldNames.KEY_PHRASES]))
        
        #auths = line_json[FieldNames.AUTHORS]

        yield ProtoDoc(
            id=line_json[FieldNames.PAPER_ID],
            title=line_json[FieldNames.TITLE],
            abstract=line_json[FieldNames.ABSTRACT],
            out_citations=citations,
            in_citation_count=in_citation_count,
            year=line_json.get(FieldNames.YEAR, 2017),
            key_phrases=key_phrases,
            venue=line_json.get(FieldNames.VENUE, ''),
        )


def build_corpus(db_filename, corpus_json):
    """"""
    with sqlite3.connect(db_filename) as conn:
        conn.execute('PRAGMA synchronous=OFF')
        conn.execute('PRAGMA journal_mode=MEMORY')
        conn.row_factory = sqlite3.Row
        conn.execute(
            '''CREATE TABLE IF NOT EXISTS ids (key INTEGER PRIMARY KEY ASC, id STRING, year INT)'''
        )
        conn.execute(
            '''CREATE TABLE IF NOT EXISTS documents
                    (key INTEGER PRIMARY KEY ASC, id STRING, year INT, payload BLOB)'''
        )
        conn.execute('''CREATE INDEX IF NOT EXISTS year_idx on ids (year)''')
        conn.execute('''CREATE INDEX IF NOT EXISTS id_idx on ids (id)''')
        conn.execute('''CREATE INDEX IF NOT EXISTS id_doc_idx on documents (id)''')
        
        for file in corpus_json:
            for batch in batchify(stream_papers(file), 1024):
                conn.executemany(
                    'INSERT INTO ids (id, year) VALUES (?, ?)',
                    [
                        (doc.id, doc.year)
                        for doc in batch
                    ]
                )
                conn.executemany(
                    'INSERT INTO documents (id, payload) VALUES (?, ?)',
                    [
                        (doc.id, doc.SerializeToString())
                        for doc in batch
                    ]
                )

        conn.commit()


def load(data_path, train_frac=0.80):
    return Corpus(data_path, train_frac)

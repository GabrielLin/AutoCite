#!/usr/bin/env python
"""
Luigi pipeline for Citeomatic.

This includes tasks for fetching the dataset, building a vocabulary and
training features and training/evaluating the model.
"""
import logging
import os
import zipfile
from os import path

import luigi
from citeomatic import file_util, features, training, corpus
from citeomatic.features import Featurizer
from citeomatic.models.options import ModelOptions
from citeomatic.serialization import import_from
from luigi.util import inherits

logger = logging.getLogger('citeomatic.tasks')

import faulthandler
faulthandler.enable()


class SharedParameters(luigi.Task):
    base_dir = luigi.Parameter(default=path.expanduser('~/citeomatic-data/'))

    @property
    def data_dir(self):
        return self.base_dir + '/data'

    @property
    def model_dir(self):
        return self.base_dir + '/model'

    def log(self, msg, *args):
        logger.info(msg, *args)


class DownloadCorpus(SharedParameters):
    corpus_url = luigi.Parameter(
        default=
        'https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/2017-02-21/papers-2017-02-21.zip'
    )

    def output(self):
        json_name = self.corpus_url.split('/')[-1]
        json_name = json_name.replace('.zip', '.json.gz')
        return luigi.LocalTarget(path.join(self.data_dir, json_name))

    def run(self):
        self.output().makedirs()

        output_dir = path.dirname(self.output().path)
        output_filename = self.output().path

        assert os.system(
            'curl "%s" > "%s/papers.zip.tmp"' % (self.corpus_url, output_dir)
        ) == 0

        with zipfile.ZipFile('%s/papers.zip.tmp' % output_dir) as zf:
            for name in zf.namelist():
                if name.endswith('.json.gz'):
                    zf.extract(name, output_dir)
                    break

        #assert os.unlink('%s/papers.zip.tmp' % output_dir) == 0


class BuildCorpus(SharedParameters):
    def requires(self):
        return {'corpus': DownloadCorpus()}

    def output(self):
        corpus_suffix = self.requires()['corpus'].corpus_url.split('/')[-1]
        corpus_name = corpus_suffix.replace('.zip', '.sqlite')
        return luigi.LocalTarget(path.join(self.data_dir, corpus_name))

    def run(self):
        try:
            corpus.build_corpus(self.output().path + '.tmp', self.input()['corpus'].path)
            os.rename(self.output().path + '.tmp', self.output().path)
        except:
            os.system("rm -rf '%s'" % self.output().path + '.tmp')
            raise


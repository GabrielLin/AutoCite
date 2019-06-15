import argparse
import logging
import os
import pickle
import sys
import time
import typing
from ast import literal_eval

import numpy
import pandas
import traitlets
from traitlets.config import Configurable

from citeomatic import traits, file_util
from .file_util import read_json, read_pickle, write_file, write_json, write_pickle

# These properties will be ignored for argument parsing.
IGNORED_TRAITS = {'parent', 'config'}


def generic_parser(trait, v):
    if v.startswith('@json:'):
        try:
            return read_json(v[6:])
        except Exception as e:
            raise argparse.ArgumentTypeError('Failed to parse JSON', e)

    if v.startswith('@eval:'):
        try:
            return eval(v[6:])
        except Exception as e:
            raise argparse.ArgumentTypeError('Failed to evaluate argument', e)

    if v.startswith('@pickle:'):
        try:
            return read_pickle(v[8:])
        except Exception as e:
            raise argparse.ArgumentTypeError(
                'Failed to read pickle file %s' % v[8:], e
            )

    if v.startswith('@csv:'):
        try:
            return pandas.read_csv(v[5:])
        except Exception as e:
            raise argparse.ArgumentTypeError(
                'Failed to read CSV file %s' % v[5:], e
            )

    if v.startswith('@call:'):
        try:
            import importlib
            fqn = v[6:]
            module_parts = fqn.split('.')
            module_name = '.'.join(module_parts[:-1])
            fn = module_parts[-1]
            mod = importlib.import_module(module_name)
            return getattr(mod, fn)()
        except Exception as e:
            raise argparse.ArgumentTypeError(
                'Failed to invoke method: %s: %s' % (v, e)
            )

    if isinstance(trait, (traitlets.Unicode, traitlets.Enum)):
        return v

    if isinstance(trait, traitlets.Int):
        return int(v)

    if isinstance(trait, traitlets.Bool):
        try:
            iv = int(v)
            return bool(iv)
        except ValueError as _:
            pass

        if v.lower() == 'true':
            return True
        if v.lower() == 'false':
            return False
        raise argparse.ArgumentTypeError(
            '"%s" could not be parsed as a boolean'
        )

    return literal_eval(v)


def parser_for_trait(trait: traitlets.TraitType) -> object:
    def _trait_parser(v):
        return generic_parser(trait, v)

    _trait_parser.__name__ = trait.__class__.__name__
    return _trait_parser


def setup_default_logging(level=logging.INFO):
    pandas.options.display.width = 200
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)

    logging.basicConfig(
        format=
        '%(levelname).1s%(asctime)-15s %(filename)s:%(lineno)d %(message)s',
        level=level,
        stream=sys.stderr
    )

    logging.getLogger('elasticsearch').setLevel(logging.WARN)



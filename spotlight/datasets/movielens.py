"""
Utilities for fetching the Movielens datasets [1]_.

References
----------

.. [1] https://grouplens.org/datasets/movielens/
"""

import os
import pandas as pd
import h5py

from spotlight.datasets import _transport
from spotlight.interactions import Interactions

VARIANTS = ('100K',
            '1M',
            '10M',
            '20M')


URL_PREFIX = ('https://github.com/maciejkula/recommender_datasets/'
              'releases/download')
VERSION = 'v0.2.0'


def _get_movielens(dataset):
    print(dataset)

    extension = '.hdf5'
    path = 'datasets/movielens/' + dataset + extension


    # path = _transport.get_data('/'.join((URL_PREFIX,
    #                                      VERSION,
    #                                      dataset + extension)),
    #                            os.path.join('movielens', VERSION),
    #                            'movielens_{}{}'.format(dataset,
    #                                                    extension))

    with h5py.File(path, 'r') as data:
        return (data['/user_id'][:],
                data['/item_id'][:],
                data['/rating'][:],
                data['/timestamp'][:])

def get_local_variant(variant):
    train_data = pd.read_csv('ml-20m/pro_sg/validation_tr.csv')
    test_data_tr = pd.read_csv('ml-20m/pro_sg/test_tr.csv')
    users_train = train_data.uid.values
    movies_train = train_data.sid.values

def get_movielens_dataset(variant='100K'):
    """
    Download and return one of the Movielens datasets.

    Parameters
    ----------

    variant: string, optional
         String specifying which of the Movielens datasets
         to download. One of ('100K', '1M', '10M', '20M').

    Returns
    -------

    Interactions: :class:`spotlight.interactions.Interactions`
        instance of the interactions class
    """

    if variant not in VARIANTS:
        raise ValueError('Variant must be one of {}, '
                         'got {}.'.format(VARIANTS, variant))

    url = 'movielens_{}'.format(variant)
    users,items,ratings,timestamps =(_get_movielens(url))
    return Interactions(users,items,ratings,timestamps)

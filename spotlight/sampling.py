"""
Module containing functions for negative item sampling.
"""

import numpy as np
import logging
import time

def sample_items(interaction,user_ids,num_items, shape, random_state=None):
    """
    Randomly sample a number of items.

    Parameters
    ----------

    num_items: int
        Total number of items from which we should sample:
        the maximum value of a sampled item id will be smaller
        than this.
    shape: int or tuple of ints
        Shape of the sampled array.
    random_state: np.random.RandomState instance, optional
        Random state to use for sampling.

    Returns
    -------

    items: np.array of shape [shape]
        Sampled item ids.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    items = random_state.randint(0, num_items, shape, dtype=np.int64)

    # for i,j in enumerate(user_ids.cpu().data):
    #     if(interaction.has_key(j.item(),items[i])):
    #         items[i] = negsamp_vectorized_bsearch_preverif(interaction.tocsr()[j.item(),:].toarray().nonzero()[1],interaction.num_items,1)[0] 
    return items

def negsamp_vectorized_bsearch_preverif(pos_inds, n_items, n_samp=32):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds

def get_negative_samples(train,num_samples):
    interactions = train.tocsr()
    num_items = train.num_items
    neg_samples = []
    logging.info("Generating %d Samples"%num_samples)
    start = time.time()

    users = np.random.choice(np.unique(train.user_ids),num_samples)

    for user in users:
        item = negsamp_vectorized_bsearch_preverif (interactions[user,:].toarray().nonzero()[1],num_items,1)[0]
        pair = (user,item)
        neg_samples.append(pair)

    end = time.time()
    logging.info("Took %d seconds"%(end - start))
    return neg_samples



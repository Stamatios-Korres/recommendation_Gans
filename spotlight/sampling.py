"""
Module containing functions for negative item sampling.
"""

import numpy as np


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
    #Shape reflects the number of uses for whom to select negative items
    
    # items = random_state.randint(0, num_items, shape, dtype=np.int64)

    # Very slow implemenation 

    array = np.zeros((shape,))
    i=0
    
    for u in user_ids.cpu().data:
        j = np.random.randint(interaction.num_items)
        while interaction.has_key(u, j):
            j = np.random.randint(interaction.num_items)
        array[i] = j
        i+=1

    # if random_state is None:
    #     random_state = np.random.RandomState()
    # items = random_state.randint(0, num_items, shape, dtype=np.int64)
    return array

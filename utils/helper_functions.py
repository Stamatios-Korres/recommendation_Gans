import torch
import numpy as np
from spotlight.interactions import Interactions



def make_implicit(interactions):
    """
    Parameters
    ----------

    Interactions: :class:`spotlight.interactions.Interactions`
    instance of the interactions class
    Returns
    -------

    Interactions: :class:`spotlight.interactions.Interactions`
    instance of the interactions class after making ratings implicit
    """
    
    ratings = interactions.ratings
    ratings = np.array([1 if x > 3.5  else 0 for x in ratings])
    interactions.ratings = ratings

    return interactions


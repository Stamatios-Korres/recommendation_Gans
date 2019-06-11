
import numpy as np
import torch
import torch.optim as optim
import random
import logging
import tqdm
import copy

class CGAN_slateGeneration(object):

    """
    Class representing generative model for direct slate generation
    using conditional adversarial networks (cGANs)


    Parameters 
    ----------
    G: nn.module
        The architecture of the generator. It will be used to directly generate slates,
        in order to learn the distribution of a given user. The user embedding will be the condition.
    Must output the size of the slate (k)
    D: nn.module
        Discriminator which will decide, given a user, whether the slate is real or fale. 
        Input: k-size np.array representing the slate
        Ouput: {0,1} whether slate is real or not.
    n_iter: int, optional
        Number of iterations to run.
    batch_size: int, optional
        Minibatch size.
    l2: float, optional
        L2 loss penalty.
    learning_rate: float, optional
        Initial learning rate.
    optimizer_func: function, optional
        Function that takes in module parameters as the first argument and
        returns an instance of a PyTorch optimizer. Overrides l2 and learning
        rate if supplied. If no optimizer supplied, then use ADAM by default.
    use_cuda: boolean, optional
        Run the model on a GPU.
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    random_state: instance of numpy.random.RandomState, optional
        Random state to use when fitting.
    """

    def __init__(self,G
                     ,D
                     ,n_iter,
                     batcj_size,
                     l2,
                     learning_rate,
                     optimizer_func,
                     use_cuda,
                     sparse,
                     random_state):
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._use_cuda = use_cuda
        self._sparse = sparse
        self._optimizer_func = optimizer_func
        self._random_state = random_state or np.random.RandomState()


        self._num_users = None
        self._num_items = None
        self._net = None
        self._optimizer = None
        self._loss_func = None
        self.best_model = None
        self.best_validation = None

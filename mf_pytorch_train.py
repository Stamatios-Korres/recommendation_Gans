
from mf_pytorch import BasePipeline
import torch
import mf_pytorch
import numpy as np

def train_test_split(interactions, n=10):
    """
    Split an interactions matrix into training and test sets.
    Parameters
    ----------
    interactions : np.ndarray
    n : int (default=10)
        Number of items to select / row to place into test.
    Returns
    -------
    train : np.ndarray
    test : np.ndarray
    """
    test = np.zeros(interactions.shape)
    train = interactions.toarray()
    # interactions.copy()
    # interactions.toarray()
    for user in range(interactions.shape[0]):
        if interactions[user, :].nonzero()[0].shape[0] > n:
            test_interactions = np.random.choice(interactions[user, :].nonzero()[0],
                                                 size=n,
                                                 replace=False)
            train[user, test_interactions] = 0.
            test[user, test_interactions] = interactions.toarray()[user, test_interactions]

    # Test and training are truly disjoint
    assert(np.all((train * test) == 0))
    return train, test


def implicit(interactions):
    train, test = train_test_split(interactions)

    pipeline = BasePipeline(train, test=test, verbose=True,
                           batch_size=1024, num_workers=4,
                           n_factors=20, weight_decay=0,
                           dropout_p=0., lr=.2, sparse=True,
                           optimizer=torch.optim.SGD, n_epochs=40,
                           random_seed=2017, loss_function=mf_pytorch.bpr_loss,
                           model=mf_pytorch.BPRModule,
                           interaction_class=mf_pytorch.PairwiseInteractions,
                           eval_metrics=('auc', 'patk'))
    pipeline.fit()
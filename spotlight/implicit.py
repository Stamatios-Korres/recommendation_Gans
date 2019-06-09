"""
Factorization models for implicit feedback problems.
"""

import numpy as np
import torch
import torch.optim as optim
import random
import logging
import tqdm
import copy


from spotlight.helpers import _repr_model
from spotlight.factorization._components import _predict_process_ids
from spotlight.losses import (adaptive_hinge_loss,
                              bpr_loss,
                              hinge_loss,
                              pointwise_loss)
from spotlight.factorization.representations import BilinearNet
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle
from spotlight.evaluation import rmse_score,precision_recall_score,evaluate_popItems,evaluate_random

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format='%(message)s',level=logging.INFO)


class ImplicitFactorizationModel(object):
    """
    An implicit feedback matrix factorization model. Uses a classic
    matrix factorization [1]_ approach, with latent vectors used
    to represent both users and items. Their dot product gives the
    predicted score for a user-item pair.

    The latent representation is given by
    :class:`spotlight.factorization.representations.BilinearNet`.

    The model is trained through negative sampling: for any known
    user-item pair, one or more items are randomly sampled to act
    as negatives (expressing a lack of preference by the user for
    the sampled item).

    .. [1] Koren, Yehuda, Robert Bell, and Chris Volinsky.
       "Matrix factorization techniques for recommender systems."
       Computer 42.8 (2009).

    Parameters
    ----------

    loss: string, optional
        One of 'pointwise', 'bpr', 'hinge', or 'adaptive hinge',
        corresponding to losses from :class:`spotlight.losses`.
    embedding_dim: int, optional
        Number of embedding dimensions to use for users and items.
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
    representation: a representation module, optional
        If supplied, will override default settings and be used as the
        main network module in the model. Intended to be used as an escape
        hatch when you want to reuse the model's training functions but
        want full freedom to specify your network topology.
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    random_state: instance of numpy.random.RandomState, optional
        Random state to use when fitting.
    num_negative_samples: int, optional
        Number of negative samples to generate for adaptive hinge loss.
    """

    def __init__(self,
                 loss='pointwise',
                 embedding_dim=32,
                 n_iter=10,
                 batch_size=256,
                 l2=0.0,
                 learning_rate=1e-2,
                 optimizer_func=None,
                 use_cuda=False,
                 representation=None,
                 sparse=False,
                 random_state=None,
                 neg_examples=None,
                 num_negative_samples=3):

        assert loss in ('pointwise',
                        'bpr',
                        'hinge',
                        'adaptive_hinge')


        self._loss = loss
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda
        self._representation = representation
        self._sparse = sparse
        self._optimizer_func = optimizer_func
        self._random_state = random_state or np.random.RandomState()

        self._num_negative_samples = num_negative_samples
        self.neg_examples = neg_examples 
        
        self._num_users = None
        self._num_items = None
        self._net = None
        self._optimizer = None
        self._loss_func = None
        self.neg_examples = neg_examples 
        self.best_model = None
        self.best_validation = None

        self._writer = SummaryWriter()
        # log_dir='runs/{}'.format('experiment'))

        set_seed(self._random_state.randint(-10**8, 10**8),
                 cuda=self._use_cuda)

    def __repr__(self):

        return _repr_model(self)

    @property
    def _initialized(self):
        return self._net is not None


    def set_users(self,_num_users,_num_items):
        self._num_users = _num_users
        self._num_items = _num_items
        self._net = gpu(self._representation,
                            self._use_cuda)

    def _initialize(self, interactions):

        (self._num_users,
         self._num_items) = (interactions.num_users,
                             interactions.num_items)

        if self._representation is not None:
            self._net = gpu(self._representation,
                            self._use_cuda)
        else:
            self._net = gpu(
                BilinearNet(self._num_users,
                            self._num_items,
                            self._embedding_dim,
                            sparse=self._sparse),
                self._use_cuda
            )

        if self._optimizer_func is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
        else:
            self._optimizer = self._optimizer_func(
                self._net.parameters(), 
                weight_decay=self._l2,
                lr=self._learning_rate)

        if self._loss == 'pointwise':
            self._loss_func = pointwise_loss
        elif self._loss == 'bpr':
            self._loss_func = bpr_loss
        elif self._loss == 'hinge':
            self._loss_func = hinge_loss
        else:
            self._loss_func = adaptive_hinge_loss

    def _check_input(self, user_ids, item_ids, allow_items_none=False):

        if isinstance(user_ids, int):
            user_id_max = user_ids
        else:
            user_id_max = user_ids.max()

        if user_id_max >= self._num_users:
            raise ValueError('Maximum user id greater '
                             'than number of users in model.')

        if allow_items_none and item_ids is None:
            return

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._num_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')

    def fit(self, train_set , valid_set, verbose=False):

        
        """
        Fit the model.

        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------

        interactions: :class:`spotlight.interactions.Interactions`
            The input dataset.

        verbose: bool
            Output additional information about current epoch and loss.
        """
        

        user_ids = train_set.user_ids
        item_ids = train_set.item_ids
        
        self.ratio  = len(train_set)/(train_set.num_items*train_set.num_users)
        
        
        users, items = shuffle(user_ids, item_ids, random_state=self._random_state)
           
        user_ids_tensor = gpu(torch.from_numpy(users), self._use_cuda).long()
        item_ids_tensor = gpu(torch.from_numpy(items), self._use_cuda).long()

        user_ids_valid_tensor = gpu(torch.from_numpy(valid_set.user_ids), self._use_cuda).long()
        item_ids_valid_tensor = gpu(torch.from_numpy(valid_set.item_ids), self._use_cuda).long()


        if not self._initialized:
            self._initialize(train_set)

        self._check_input(user_ids, item_ids)

        for epoch_num in range(self._n_iter):

           

            train_epoch_loss = 0.0
            valid_epoch_loss = 0.0

            #Train Model
            self._net.train()

            with tqdm.tqdm(total=len(train_set)) as pbar_train:
                for (minibatch_num,(batch_user,  batch_item)) in enumerate(minibatch(user_ids_tensor, item_ids_tensor,batch_size=self._batch_size)):
            
                    loss = self.run_train_iteration(batch_user,batch_item)

                    train_epoch_loss += loss.item()
                  
       
                    pbar_train.update(self._batch_size)
                    pbar_train.set_description("loss: {:.4f}".format(loss.item()))
            
            train_epoch_loss /= minibatch_num + 1

            if np.isnan(train_epoch_loss) or train_epoch_loss == 0.0:
                    raise ValueError('Degenerate epoch loss: {}'
                                     .format(train_epoch_loss))
          

            self._writer.add_scalar('training_loss', train_epoch_loss, epoch_num)
            

            #Validate Model
            self._net.eval()

            with tqdm.tqdm(total=len(valid_set)) as pbar_val:
                for (minibatch_num,(batch_user,  batch_item)) in enumerate(minibatch(user_ids_valid_tensor, item_ids_valid_tensor,batch_size=self._batch_size)):
                
                    val_loss = self.run_val_iteration(batch_user,batch_item)
                    valid_epoch_loss += val_loss.item()
                    pbar_val.update(self._batch_size)
                    pbar_val.set_description("loss: {:.4f}".format(val_loss.item()))
                    
            
                valid_epoch_loss /= minibatch_num + 1
                self._writer.add_scalar('validation_loss', valid_epoch_loss, epoch_num)
                if self.best_validation == None or self.best_validation > valid_epoch_loss:
                    self.best_model = copy.deepcopy(self._net)
                    self.best_validation = valid_epoch_loss
            if verbose:
                logging.info('Epoch {}: training_loss {:10.6f}'.format(epoch_num, train_epoch_loss))
                logging.info('Epoch {}: validation_loss {:10.6f}'.format(epoch_num, valid_epoch_loss))

        self._net = self.best_model                         
        self._writer.close()          

    def run_train_iteration(self,batch_user, batch_item):
        positive_prediction = self._net(batch_user, batch_item)
        self._optimizer.zero_grad()

        if self.neg_examples:
            user_neg_ids,item_neg_ids = zip(*random.choices(self.neg_examples, k = self._batch_size ))
            user_neg_ids_tensor = gpu(torch.from_numpy(np.array(user_neg_ids)),
                        self._use_cuda).long()
            item_neg_ids_tensor = gpu(torch.from_numpy(np.array(item_neg_ids)),
                        self._use_cuda).long()

            negative_prediction = self._net(user_neg_ids_tensor, item_neg_ids_tensor)
            loss = self._loss_func(positive_prediction,negative_prediction,ratio=self.ratio)
        else:
            loss = self._loss_func(positive_prediction)
        loss.backward()
        self._optimizer.step()

        return loss

    def run_val_iteration(self,batch_user,batch_item):
        
        positive_prediction = self._net(batch_user, batch_item)
        if self.neg_examples:
            user_neg_ids,item_neg_ids = zip(*random.choices(self.neg_examples, k = self._batch_size ))
            user_neg_ids_tensor = gpu(torch.from_numpy(np.array(user_neg_ids)),
                        self._use_cuda).long()
            item_neg_ids_tensor = gpu(torch.from_numpy(np.array(item_neg_ids)),
                        self._use_cuda).long()
            negative_prediction = self._net(user_neg_ids_tensor, item_neg_ids_tensor)
            loss = self._loss_func(positive_prediction,negative_prediction,ratio=self.ratio)
        else:
            loss = self._loss_func(positive_prediction)
        return loss

    def predict(self, user_ids, item_ids=None):

        """
        Make predictions: given a user id, compute the recommendation
        scores for items.

        Parameters
        ----------

        user_ids: int or array
           If int, will predict the recommendation scores for this
           user for all items in item_ids. If an array, will predict
           scores for all (user, item) pairs defined by user_ids and
           item_ids.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.

        Returns
        -------

        predictions: np.array
            Predicted scores for all items in item_ids.
        """
 
        self._check_input(user_ids, item_ids, allow_items_none=True)
        self._net.train(False)

        user_ids, item_ids = _predict_process_ids(user_ids, item_ids, self._num_items, self._use_cuda)

        out = self._net(user_ids, item_ids)

        return cpu(out).detach().numpy().flatten()

    def test(self,test_set,item_popularity,k=5):

        user_ids_valid_tensor = gpu(torch.from_numpy(test_set.user_ids), self._use_cuda).long()
        item_ids_valid_tensor = gpu(torch.from_numpy(test_set.item_ids), self._use_cuda).long()

        rmse_test_loss = 0 

        for (minibatch_num,(batch_user,  batch_item)) in enumerate(minibatch(user_ids_valid_tensor, item_ids_valid_tensor,batch_size=self._batch_size)):
    
            loss = rmse_score(self._net,batch_user, batch_item)
            rmse_test_loss += loss
        
        rmse_test_loss /= minibatch_num + 1

        # rmse = rmse_score(self, test)

        logging.info("RMSE: {}".format(rmse_test_loss))

        pop_precision,pop_recall = evaluate_popItems(item_popularity,test_set,k=k)
        rand_precision, rand_recall = evaluate_random(item_popularity,test_set,k=k)

        precision,recall = precision_recall_score(self,test=test_set,k=k)

        # self._writer.add_scalar('precision', precision, epoch_num)
        # self._writer.add_scalar('recall', recall, epoch_num)


        logging.info("Random: precision {} recall {}".format(rand_precision,rand_recall))
        logging.info("PopItem Algorithm: precision {} recall {}".format(pop_precision,pop_recall))
        logging.info("My model: precision {} recall {}".format(precision,recall))


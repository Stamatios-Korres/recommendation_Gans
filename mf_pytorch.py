
# Headers and imports
import numpy as np 
import pandas as pd 
import mf
from nfc.src.mlp import MLP as mlp
from scipy.sparse import csr_matrix
from mf import MF 
import os
import sys
from pytorch.torchmf import BasePipeline,bpr_loss,BPRModule,PairwiseInteractions, Interactions, BaseModule
from scipy.sparse import coo_matrix
import torch
from pytorch.metrics import precision_at_k

from pytorch import utils

train_data = pd.read_csv('ml-20m/pro_sg/validation_tr.csv')
test_data_tr = pd.read_csv('ml-20m/pro_sg/test_tr.csv')
users = 136677 
movies = 20720


users_train = train_data.uid.values
movies_train = train_data.sid.values
data_train = np.ones((users_train.shape[0]))
coo_train = coo_matrix((data_train, (users_train,movies_train)),shape=(users, movies))

#Dimensions of the coo_matrix (Users x Movies)
n_test_users = test_data_tr.uid.unique()
n_test_movies =  test_data_tr.sid.unique()

users_test = test_data_tr.uid.values
movies_test = test_data_tr.sid.values
data_test = np.ones((users_test.shape[0]))

coo_test = coo_matrix((data_test, (users_test,movies_test)),shape=(users, movies))

train, test = utils.get_movielens_train_test_split(implicit=True)

print(type(train),type(test),train.shape,test.shape)
# print(type(coo_train),type(coo_test),coo_train.shape,coo_test.shape)

pipeline = BasePipeline(train= train, test=None, verbose=True,
                        batch_size=1024, num_workers=1,
                        n_factors=15, weight_decay=0,
                        dropout_p=0., lr=.01, sparse=True,
                        optimizer=torch.optim.SGD, n_epochs=40,
                        random_seed=2017, loss_function=bpr_loss,
                        model=BPRModule, hogwild=False,
                        interaction_class=PairwiseInteractions,
                        eval_metrics=['precision_at_k']
                    )
                        
pipeline.fit()
import torch.utils.data as data
test_dataloader = data.DataLoader( PairwiseInteractions(test), batch_size=1024, shuffle=True)
res = precision_at_k(pipeline.model, test_dataloader.dataset.mat_csr)


torch.save(pipeline.model, 'mf_model')
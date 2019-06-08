import torch
import numpy as np
from spotlight.cross_validation import train_test_timebased_split
from spotlight.datasets.movielens import get_movielens_dataset

import spotlight.optimizers as optimizers
from spotlight.factorization.representations import BilinearNet
from spotlight.implicit import ImplicitFactorizationModel
from spotlight.sampling import get_negative_samples
from utils.helper_functions import make_implicit
from utils.arg_extractor import get_args
from spotlight.nfc.mlp import MLP as mlp


import logging
logging.basicConfig(format='%(message)s',level=logging.INFO)


args = get_args()  # get arguments from command line
assert (args.optim in ('sgd,adam'))
use_cuda=args.use_gpu
dataset_name = args.dataset


logging.info("DataSet MovieLens_%s will be used"%dataset_name)



if args.on_cluster:
    path = '/disk/scratch/s1877727/datasets/movielens/'
else:
    path = 'datasets/movielens/'

dataset,item_popularity = get_movielens_dataset(variant=dataset_name,path=path)

# ------------------- #
#Transform the dataset to implicit feedback

dataset = make_implicit(dataset)



train,test = train_test_timebased_split(dataset,test_percentage=0.2)
train,valid = train_test_timebased_split(train,test_percentage=0.2)


logging.info("Creating random negative examples from train set")


#Training parameters
users, movies = train.num_users,train.num_items
embedding_dim = args.embedding_dim
training_epochs = args.training_epochs
learning_rate = args.learning_rate
l2_regularizer = args.l2_regularizer
batch_size = args.batch_size
neg_examples = get_negative_samples(train,(train.__len__())*args.neg_examples)

# Choose training model
if args.model == 'mlp':
    layers = [16, 8, 4]
    technique = mlp(layers=layers,num_users=users,num_items=movies,embedding_dim = embedding_dim)
else:
    technique = BilinearNet(users, movies, embedding_dim, sparse=False)

# Choose optimizer 
optim = getattr(optimizers, args.optim + '_optimizer')

# Print statistics of the experiment
logging.info("Training session: {} latent dimensions, {} epochs, {} batch size {} learning rate.  {} users x  {} items".format(embedding_dim, training_epochs,batch_size,learning_rate,users,movies))
logging.info("Training interaction: {} Test interactions, {}".format(train.__len__(),test.__len__()))

#Initialize model
model = ImplicitFactorizationModel( n_iter=training_epochs,neg_examples = None,
                                    embedding_dim=embedding_dim,l2=l2_regularizer,
                                    representation=technique,
                                    batch_size = batch_size,use_cuda=use_cuda,learning_rate=learning_rate,
                                    optimizer_func=optim)

logging.info("Model set, training begins")

model.fit(train,valid,verbose=True)

logging.info("Model is ready, testing performance")

network = model._net

torch.save(network.state_dict(), args.experiment_name)

model.test(test,item_popularity,args.k)


# python3 mf_spotlight.py --model mlp --embedding_dim 8  --learning_rate 1e-3 --l2_regularizer 1e-7 --training_epochs 100
# python3 mf_spotlight.py --model mlp --embedding_dim 8  --learning_rate 1e-3 --l2_regularizer 1e-6 --training_epochs 50 My model: precision 0.23584229390681002 recall 0.03213645492312779

# python3 mf_spotlight.py --model mlp --embedding_dim 8 --learning_rate 1e-3 --l2_regularizer 1e-5 --training_epochs 60 precision 0.2057347670250896 recall 0.03673811759117582


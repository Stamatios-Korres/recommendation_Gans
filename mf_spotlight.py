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
from utils.data_provider import data_provider
import math

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

data_loader  = data_provider(path,dataset_name,args.neg_examples)
train,valid,test,neg_examples,item_popularity = data_loader.get_data()

#Reproducability of results 
seed = 0 
random_state = np.random.RandomState(seed) 
torch.manual_seed(seed)


#Training parameters
users, movies = train.num_users,train.num_items
embedding_dim = args.embedding_dim
training_epochs = args.training_epochs
learning_rate = args.learning_rate
l2_regularizer = args.l2_regularizer
batch_size = args.batch_size

# Choose training model
if args.model == 'mlp':
    top = math.log2(embedding_dim*2)
    layers = [2**x for x in reversed(range(3,int(top)+1))] 
    technique = mlp(layers=layers,num_users=users,num_items=movies,embedding_dim = embedding_dim)
else:
    technique = BilinearNet(users, movies, embedding_dim, sparse=False)

# Choose optimizer 
optim = getattr(optimizers, args.optim + '_optimizer')

# Print statistics of the experiment
logging.info("Training session: {} latent dimensions, {} epochs, {} batch size {} learning rate.  {} users x  {} items".format(embedding_dim, training_epochs,batch_size,learning_rate,users,movies))
logging.info("Training interaction: {} Test interactions, {}".format(train.__len__(),test.__len__()))

#Initialize model
model = ImplicitFactorizationModel( n_iter=training_epochs,neg_examples = neg_examples,
                                    num_negative_samples = args.neg_examples,
                                    embedding_dim=embedding_dim,l2=l2_regularizer,
                                    representation=technique,random_state=random_state,
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
# python3 mf_spotlight.py --model mlp --embedding_dim 8 --learning_rate 3e-3 --l2_regularizer 1e-5 --training_epochs 30 --batch_size 256
# python mf_spotlight.py --embedding_dim 100 --training_epochs 80 --learning_rate 0.001 --l2_regularizer 1e-3 --k 5  precision 0.26594982078853047 recall 0.03786741692918603
# python3 mf_spotlight.py --model mlp --embedding_dim 8 --learning_rate 3e-3 --l2_regularizer 1e-5 --training_epochs 30 --batch_size 256 My model: precision 0.21935483870967742 recall 0.025725807258139947


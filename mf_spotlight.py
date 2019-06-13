import torch
import numpy as np
from spotlight.dataset_manilupation import train_test_timebased_split
from spotlight.datasets.movielens import get_movielens_dataset
import spotlight.optimizers as optimizers
from spotlight.factorization.representations import BilinearNet
from implicit import ImplicitFactorizationModel
from spotlight.sampling import get_negative_samples
from utils.arg_extractor import get_args
from spotlight.Dnn_models.mlp import MLP as mlp
from spotlight.Dnn_models.neuMF import NeuMF as neuMF
from utils.data_provider import data_provider
import math


import logging

logging.basicConfig(format='%(message)s',level=logging.INFO)

args = get_args()  # get arguments from command line
use_cuda=args.use_gpu
dataset_name = args.dataset

logging.info("DataSet MovieLens_%s will be used"%dataset_name)

if args.on_cluster:
    path = '/disk/scratch/s1877727/datasets/movielens/'
else:
    path = 'datasets/movielens/'

rmse_flag = args.rmse
pre_recall_flag = args.precision_recall
map_recall_flag= args.map_recall

print(rmse_flag ,pre_recall_flag ,map_recall_flag )
#Reproducability of results 
seed = 0 
random_state = np.random.RandomState(seed) 
torch.manual_seed(seed)

# Get data for train and test
data_loader  = data_provider(path,dataset_name,args.neg_examples)
train,valid,test,neg_examples,item_popularity = data_loader.get_timebased_data()

#Training parameters
users, movies = train.num_users,train.num_items
training_epochs = args.training_epochs
learning_rate = args.learning_rate
l2_regularizer = args.l2_regularizer
batch_size = args.batch_size


# Choose training model
if args.model == 'mlp':
    model_name = 'mlp'
    mlp_embedding_dim = args.mlp_embedding_dim
    top = math.log2(mlp_embedding_dim*2)
    mlp_layers = [2**x for x in reversed(range(3,int(top)+1))] 
    technique = mlp(layers=mlp_layers,num_users=users,num_items=movies,embedding_dim = mlp_embedding_dim)
elif args.model == 'mf':
    model_name = 'mf'
    mf_embedding_dim = args.mf_embedding_dim
    technique = BilinearNet(users, movies, mf_embedding_dim, sparse=False)
elif args.model == 'neuMF':
    model_name = 'neuMF'
    mf_embedding_dim = args.mf_embedding_dim
    mlp_embedding_dim = args.mlp_embedding_dim
    top = math.log2(mlp_embedding_dim*2)
    mlp_layers = [2**x for x in reversed(range(3,int(top)+1))] 
    technique = neuMF(mlp_layers=mlp_layers,num_users= users, num_items= movies, mf_embedding_dim=mf_embedding_dim,mlp_embedding_dim=mlp_embedding_dim)

# Choose optimizer 
optim = getattr(optimizers, args.optim + '_optimizer')

embedding_dim = mlp_embedding_dim if args.model == 'mlp' else mf_embedding_dim

#Initialize model
model = ImplicitFactorizationModel( n_iter=training_epochs,neg_examples = neg_examples,
                                    num_negative_samples = args.neg_examples,model_name = model_name,
                                    embedding_dim=embedding_dim,l2=l2_regularizer,
                                    representation=technique,random_state=random_state,
                                    batch_size = batch_size,use_cuda=use_cuda,learning_rate=learning_rate,
                                    optimizer_func=optim)

logging.info("Model set, training begins")
model.fit(train,valid,verbose=True)
logging.info("Model is ready, testing performance")

network = model._net
torch.save(network.state_dict(), args.experiment_name)
model.test(test,item_popularity,args.k,rmse_flag=rmse_flag, precision_recall=pre_recall_flag, map_recall=map_recall_flag)

# Print statistics of the experiment
logging.info("Training session: {} latent dimensions, {} epochs, {} batch size {} learning rate {} l2_regularizer.  {} users x  {} items".format(embedding_dim, training_epochs,batch_size,learning_rate,l2_regularizer,users,movies))

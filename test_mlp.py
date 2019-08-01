import torch
import numpy as np
import spotlight.optimizers as optimizers
import math
import logging

from spotlight.dataset_manilupation import train_test_timebased_split
from implicit import ImplicitFactorizationModel
from spotlight.sampling import get_negative_samples
from utils.arg_extractor import get_args
from spotlight.dnn_models.mlp import MLP as mlp
from spotlight.dnn_models.neuMF import NeuMF as neuMF
from utils.data_provider import data_provider


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

#Reproducability of results 
seed = 0 
random_state = np.random.RandomState(seed) 
torch.manual_seed(seed)

# Get data for train and test
data_loader  = data_provider(path,dataset_name,args.neg_examples,movies_to_keep=-1)
train,valid,test,neg_examples,item_popularity = data_loader.get_timebased_data()

#Training parameters
users, movies = train.num_users,train.num_items
training_epochs = args.training_epochs
learning_rate = args.learning_rate
l2_regularizer = args.l2_regularizer
batch_size = 256


# Choose training model


model_name = 'mlp'
mlp_embedding_dim = args.mlp_embedding_dim
top = math.log2(mlp_embedding_dim*2)
mlp_layers = [2**x for x in reversed(range(3,int(top)+1))] 
logging.info(mlp_layers)
technique = mlp(layers=mlp_layers,num_users=users,num_items=movies,embedding_dim = mlp_embedding_dim)
logging.info(technique)
# Choose optimizer 
optim = getattr(optimizers, args.optim + '_optimizer')

#Initialize model
model = ImplicitFactorizationModel( n_iter=training_epochs,neg_examples = neg_examples,
                                    num_negative_samples = args.neg_examples,model_name = model_name,
                                    embedding_dim=mlp_embedding_dim,l2=l2_regularizer,
                                    representation=technique,random_state=random_state,
                                    batch_size = batch_size,use_cuda=use_cuda,learning_rate=learning_rate,
                                    optimizer_func=optim,experiment_name=args.experiment_name)
experiment_folder  = 'experiments_results/testing_test_performance/saved_models/best_model'
model.set_users(users,movies)
state = torch.load(experiment_folder,map_location='cpu')
technique.load_state_dict(state['network'])
technique.eval()
model._net = technique


for k_loop in [1,3,5,10]:
    results = model.test(test,item_popularity,k = k_loop,rmse_flag=rmse_flag, precision_recall=pre_recall_flag, map_recall=map_recall_flag)
    print(k_loop,results['precision'],results['recall'])

# Print statistics of the experiment
logging.info("Training session: {} latent dimensions, {} epochs, {} batch size {} learning rate {} l2_regularizer.  {} users x  {} items".format(mlp_embedding_dim, training_epochs,batch_size,learning_rate,l2_regularizer,users,movies))











# if args.model == 'mlp':
#     model_name = 'mlp'
#     mlp_embedding_dim = args.mlp_embedding_dim
#     top = math.log2(mlp_embedding_dim*2)
#     mlp_layers = [2**x for x in reversed(range(3,int(top)+1))] 
#     technique = mlp(layers=mlp_layers,num_users=users,num_items=movies,embedding_dim = mlp_embedding_dim)
# elif args.model == 'neuMF':
#     model_name = 'neuMF'
#     mf_embedding_dim = args.mf_embedding_dim
#     mlp_embedding_dim = args.mlp_embedding_dim
#     top = math.log2(mlp_embedding_dim*2)
#     mlp_layers = [2**x for x in reversed(range(3,int(top)+1))] 
#     technique = neuMF(mlp_layers=mlp_layers,num_users= users, num_items= movies, mf_embedding_dim=mf_embedding_dim,mlp_embedding_dim=mlp_embedding_dim)
# embedding_dim = mlp_embedding_dim if args.model == 'mlp' else mf_embedding_dim
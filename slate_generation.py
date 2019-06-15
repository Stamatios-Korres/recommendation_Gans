import torch
import numpy as np
import logging
import spotlight.optimizers as optimizers

from CGANs import CGAN
from spotlight.dataset_manilupation import create_slates
from implicit import ImplicitFactorizationModel
from spotlight.sampling import get_negative_samples
from utils.arg_extractor import get_args
from spotlight.dnn_models.cGAN_models import generator, discriminator
from utils.data_provider import data_provider
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle




logging.basicConfig(format='%(message)s',level=logging.INFO)

args = get_args()  # get arguments from command line
use_cuda=args.use_gpu
dataset_name = args.dataset

logging.info("DataSet MovieLens_%s will be used"%dataset_name)

if args.on_cluster:
    path = '/disk/scratch/s1877727/datasets/movielens/'
else:
    path = 'datasets/movielens/'


#Reproducability of results 
seed = 0 
random_state = np.random.RandomState(seed) 
torch.manual_seed(seed)

# Get data for train and test
data_loader  = data_provider(path,dataset_name,args.neg_examples)
train,_,test,neg_examples,item_popularity = data_loader.get_timebased_data()

items_on_slates = 3
train,slates = create_slates(train,n = items_on_slates)
#Training parameters
users, movies = train.num_users,train.num_items
training_epochs = args.training_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size

rmse_flag = args.rmse
pre_recall_flag = args.precision_recall
map_recall_flag= args.map_recall


Disc = discriminator(condition_dim=movies,num_items= movies,input_dim=items_on_slates)
Gen = generator(condition_dim = movies,output_dim=items_on_slates)

# Choose optimizer 
optim = getattr(optimizers, args.optim + '_optimizer')

logging.info("Training session: {}  epochs, {} batch size {} learning rate.  {} users x  {} items".format( training_epochs,batch_size,learning_rate,users,movies))

model = CGAN(   n_iter=5,
                batch_size=batch_size,
                l2=0.0,
                slate_size = items_on_slates,
                learning_rate=learning_rate,
                use_cuda=use_cuda,
                G=Gen,
                D=Disc
               )



logging.info("Model set, training begins")
model.fit(train,slates)
logging.info("Model is ready, testing performance")

test,slates = create_slates(test,n = items_on_slates)
model.test(test,slates,item_popularity,items_on_slates,precision_recall=pre_recall_flag, map_recall=map_recall_flag)

# network = model._net
# torch.save(network.state_dict(), args.experiment_name)
# model.test(test,item_popularity,args.k)

# Print statistics of the experiment







# python3 mf_spotlight.py --model mlp --embedding_dim 8  --learning_rate 1e-3 --l2_regularizer 1e-7 --training_epochs 100
# python3 mf_spotlight.py --model mlp --embedding_dim 8  --learning_rate 1e-3 --l2_regularizer 1e-6 --training_epochs 50 My model: precision 0.23584229390681002 recall 0.03213645492312779


# python3 mf_spotlight.py --model mlp --embedding_dim 8 --learning_rate 1e-3 --l2_regularizer 1e-5 --training_epochs 60 precision 0.2057347670250896 recall 0.03673811759117582
# python3 mf_spotlight.py --model mlp --embedding_dim 8 --learning_rate 3e-3 --l2_regularizer 1e-5 --training_epochs 30 --batch_size 256
# python mf_spotlight.py --embedding_dim 100 --training_epochs 80 --learning_rate 0.001 --l2_regularizer 1e-3 --k 5  precision 0.26594982078853047 recall 0.03786741692918603
# python3 mf_spotlight.py --model mlp --embedding_dim 8 --learning_rate 3e-3 --l2_regularizer 1e-5 --training_epochs 30 --batch_size 256 My model: precision 0.21935483870967742 recall 0.025725807258139947


import torch
import numpy as np
import logging
import spotlight.optimizers as optimizers
import tqdm

from CGANs import CGAN
from utils.arg_extractor import get_args
from spotlight.dnn_models.cGAN_models import generator, discriminator
from utils.slate_data_provider import slate_data_provider
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

# Read required arguments from user inputs 

items_on_slates = args.items_on_slates 
training_epochs = args.training_epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
rmse_flag = args.rmse
pre_recall_flag = args.precision_recall
map_recall_flag= args.map_recall
loss = args.loss


# Get data for train and test
data_loader  = slate_data_provider(path,dataset_name,min_viewers=0,min_movies=10)
train_vec,train_slates,test_vec,test_set,users, movies = data_loader.get_data()

embedding_dim = args.gan_embedding_dim
hidden_layer = args.gan_hidden_layer


#Training parameters

Gen = generator(num_items = movies, embedding_dim = embedding_dim, hidden_layer = hidden_layer, output_dim=items_on_slates)
Disc = discriminator(num_items= movies, embedding_dim = embedding_dim, hidden_layers = [2*hidden_layer, hidden_layer], input_dim=items_on_slates)

# Choose optimizer 
optim = getattr(optimizers, args.optim + '_optimizer')

logging.info(" Training session: {}  epochs, {} batch size {} learning rate.  {} users x  {} items".format(training_epochs, batch_size,
                                                                                                          learning_rate,users, movies)
                                                                                                    )

model = CGAN(   n_iter=training_epochs,
                batch_size=batch_size,
                loss_fun = loss,
                slate_size = items_on_slates,
                learning_rate=learning_rate,
                use_cuda=use_cuda,
                G_optimizer_func = optim,
                D_optimizer_func = optim,
                G=Gen,
                D=Disc
               )


logging.info("Model set, training begins")
model.fit(train_vec,train_slates,users, movies)
logging.info("Model is ready, testing performance")

model.test(test_vec,test_set.tocsr(),precision_recall=pre_recall_flag, map_recall=map_recall_flag)

logging.info("Training complete")


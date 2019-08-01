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
total_movies = -1
min_movies = 0 
min_viewers = 5

# Get data for train and test
data_loader  = slate_data_provider(path,dataset_name,min_viewers=min_viewers,slate_size = args.slate_size,min_movies=min_movies,movies_to_keep=total_movies)
train_vec,train_slates,test_vec,test_set, num_users, num_movies,valid_vec,valid_cold_users,valid_set = data_loader.get_data()
cold_start_users = data_loader.get_cold_start_users()

# In general should be smaller than the dimensions of the output (Latent dimensions < Visible dimensions)
noise_dim = 100

Gen = generator(num_items = num_movies, noise_dim = noise_dim, 
                embedding_dim = args.gan_embedding_dim, 
                hidden_layer = [args.gan_hidden_layer//2,args.gan_hidden_layer], 
                output_dim=args.slate_size )

Disc = discriminator(num_items= num_movies, 
                     embedding_dim = args.gan_embedding_dim, 
                     hidden_layers = [2*args.gan_hidden_layer,args.gan_hidden_layer,args.gan_hidden_layer//2], 
                     input_dim=args.slate_size )


# Gen = generator(num_items = num_movies, noise_dim = noise_dim, 
#                 embedding_dim = args.gan_embedding_dim, 
#                 hidden_layer = [args.gan_hidden_layer], 
#                 output_dim=args.slate_size )

# Disc = discriminator(num_items= num_movies, 
#                      embedding_dim = args.gan_embedding_dim, 
#                      hidden_layers = [2*args.gan_hidden_layer], 
#                      input_dim=args.slate_size )

# Choose optimizer 
optim = getattr(optimizers, args.optim_gan + '_optimizer')

model = CGAN(   n_iter=args.training_epochs,
                z_dim = noise_dim,
                embedding_dim = args.gan_embedding_dim, 
                hidden_layer = args.gan_hidden_layer,
                batch_size=args.batch_size,
                loss_fun = args.loss,
                slate_size = args.slate_size ,
                learning_rate=args.learning_rate,
                use_cuda=use_cuda,
                experiment_name=args.experiment_name,
                G_optimizer_func = optim,
                D_optimizer_func = optim,
                G=Gen,
                D=Disc
                )


logging.info(" Training session: {}  epochs, {} batch size {} learning rate.  {} users x  {} items"
                                    .format(args.training_epochs, args.batch_size, 
                                      args.learning_rate,num_users, num_movies)
            )

logging.info("Model set, training begins")
model.fit(train_vec,train_slates, num_users, num_movies,valid_vec,valid_cold_users,valid_set)
logging.info("Model is ready, testing performance")

results = model.test(test_vec,test_set.tocsr(),cold_start_users)

logging.info('precision {} and recall {}'.format(results['precision'],results['recall']))
logging.info("Training complete")


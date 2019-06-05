
import torch
import numpy as np
from spotlight.cross_validation import random_train_test_split,train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score,precision_recall_score
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet
from utils.helper_functions import make_implicit
from utils.arg_extractor import get_args
import logging
import spotlight.optimizers as optimizers
logging.basicConfig(format='%(message)s',level=logging.INFO)

args = get_args()  # get arguments from command line
use_cuda=args.use_gpu
dataset_name = args.dataset


logging.info("DataSet MovieLens_%s will be used"%dataset_name)

if args.on_cluster:
    path = '/disk/scratch/s1877727/datasets/movielens/'
else:
    path = 'datasets/movielens/'

dataset = get_movielens_dataset(variant=dataset_name,path=path)

# ------------------- #
#Transform the dataset to implicit feedback

dataset = make_implicit(dataset)

# train, test = random_train_test_split(dataset,test_percentage=0.3)
train,test = train_test_split(dataset.tocoo().toarray(),n=args.k)
users, movies = train.num_users,train.num_items



#Training parameters

embedding_dim = args.embedding_dim
training_epochs = args.training_epochs
learning_rate = args.learning_rate
l2_regularizer = args.l2_regularizer
batch_size = args.batch_size

optim = getattr(optimizers, args.optim+'_optimizer')
logging.info("Training session: {} latent dimensions, {} epochs, {} batch size {:10.3f} learning rate.  {} users x  {} items".format(embedding_dim, training_epochs,batch_size,learning_rate,users,movies))
model = ImplicitFactorizationModel( n_iter=training_epochs,
                                    embedding_dim=embedding_dim,l2=l2_regularizer,
                                    batch_size = batch_size,use_cuda=use_cuda,learning_rate=learning_rate,
                                    optimizer_func=optim)
logging.info("Model set, training begins")

model.fit(train,verbose=True)

logging.info("Model is ready, testing performance")

network = model._net

torch.save(network.state_dict(), args.experiment_name)

rmse = rmse_score(model, test)
logging.info("RMSE: {}".format(rmse))

precision,recall = precision_recall_score(model=model,test=test,k=args.k)
logging.info("precision {} recall {}".format(np.mean(precision),np.mean(recall)))



# -------------------------- Read Model from memory ----------------- #

# network_read = BilinearNet(num_users=users,num_items=movies,embedding_dim=embedding_dim)
# network_read.load_state_dict(torch.load("matrix_model"))
# network_read.eval()
# model_read = ImplicitFactorizationModel(representation=network_read,
#                                         embedding_dim=embedding_dim,l2=l2_regularizer,
#                                         batch_size = batch_size,
#                                         learning_rate=learning_rate)
# model_read.set_users(users, movies)
# precision,recall = precision_recall_score(model=model_read,test=test,k=args.k)

# logging.info("precision %f, recall %f"%(np.mean(precision),np.mean(recall)))



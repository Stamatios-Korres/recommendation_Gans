import torch
import numpy as np
from spotlight.cross_validation import train_test_timebased_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score,precision_recall_score,evaluate_PopItems_Random
import spotlight.optimizers as optimizers
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet
from spotlight.sampling import get_negative_samples
from utils.helper_functions import make_implicit
from utils.arg_extractor import get_args

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

# train, test = random_train_test_split(dataset,test_percentage=0.3)
train,test = train_test_timebased_split(dataset,test_percentage=0.2)
users, movies = train.num_users,train.num_items

logging.info("Creating random negative examples from train set")
neg_examples = get_negative_samples(train,(train.__len__())*args.neg_examples)


#Training parameters

embedding_dim = args.embedding_dim
training_epochs = args.training_epochs
learning_rate = args.learning_rate
l2_regularizer = args.l2_regularizer
batch_size = args.batch_size

optim = getattr(optimizers, args.optim + '_optimizer')

logging.info("Training session: {} latent dimensions, {} epochs, {} batch size {} learning rate.  {} users x  {} items".format(embedding_dim, training_epochs,batch_size,learning_rate,users,movies))
logging.info("Training interaction: {} Test interactions, {}".format(train.__len__(),test.__len__()))

model = ImplicitFactorizationModel( n_iter=training_epochs,neg_examples = neg_examples,
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

pop_precision,pop_recall,rand_precision, rand_recall = evaluate_PopItems_Random(item_popularity,test,k=args.k)
precision,recall = precision_recall_score(model=model,test=test,k=args.k)

logging.info("Random: precision {} recall {}".format(rand_precision,rand_recall))
logging.info("PopItem Algorithm: precision {} recall {}".format(pop_precision,pop_recall))
logging.info("Matrix Factorization: precision {} recall {}".format(precision,recall))



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



import torch
import numpy as np
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score,precision_recall_score
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet
from utils.helper_functions import make_implicit
from utils.arg_extractor import get_args

args = get_args()  # get arguments from command line
use_cuda=args.use_gpu
dataset_name = args.dataset
dataset = get_movielens_dataset(variant=dataset_name)

# ------------------- #
#Transform the dataset to implicit feedback

dataset = make_implicit(dataset)

train, test = random_train_test_split(dataset)


users, movies = train.num_users,train.num_items
print("Data loaded, users %d and items %d" %(users,movies))

#Training parameters

embedding_dim = args.embedding_dim
training_epochs = args.training_epochs
learning_rate = args.learning_rate
l2_regularizer = args.l2_regularizer
batch_size = args.batch_size

model = ImplicitFactorizationModel( n_iter=training_epochs,
                                    embedding_dim=embedding_dim,l2=l2_regularizer,
                                    batch_size = batch_size,use_cuda=use_cuda,
                                    learning_rate=learning_rate)
print("Model set, training begins")

model.fit(train,verbose=True)

print("Model is ready, testing performance")

network = model._net

torch.save(network.state_dict(), args.experiment_name)

rmse = rmse_score(model, test)

# precision,recall = precision_recall_score(model, test)

# print("precision,recall :",np.mean(precision),np.mean(recall))




# -------------------------- Read Model from memory ----------------- #

network_read = BilinearNet(num_users=users,num_items=movies,embedding_dim=embedding_dim)
network_read.load_state_dict(torch.load("matrix_model"))
network_read.eval()
model_read = ImplicitFactorizationModel(representation=network_read,
                                        embedding_dim=embedding_dim,l2=l2_regularizer,
                                        batch_size = batch_size,
                                        learning_rate=learning_rate)
model_read.set_users(users, movies)
precision,recall = precision_recall_score(model_read, test)

print("precision,recall :",np.mean(precision),np.mean(recall))



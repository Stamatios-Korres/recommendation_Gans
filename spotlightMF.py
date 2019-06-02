import torch
import numpy as np
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score,precision_recall_score
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet


dataset = get_movielens_dataset(variant='1M')

# ------------------- #
#Transform the dataset to implicit feedback

ratings = dataset.ratings
ratings = np.array([1 if x > 3.5  else 0 for x in ratings])
dataset.ratings = ratings

train, test = random_train_test_split(dataset)





users, movies = train.num_users,train.num_items

#Training parameters
embedding_dim = 100
training_epochs = 200
learning_rate = 0.001
l2_regularizer = .0
batch_size = 1024
print("Data loaded, users %d and items %d" %(users,movies))
model = ImplicitFactorizationModel( n_iter=training_epochs,
                                    embedding_dim=embedding_dim,l2=l2_regularizer,
                                    batch_size = batch_size,
                                    learning_rate=learning_rate)
print("Model set, training begins")

model.fit(train,verbose=True)

print("Model is ready, testing performance")

network = model._net
torch.save(network.state_dict(), "matrix_model")
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



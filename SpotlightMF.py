
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score,precision_recall_score
from spotlight.factorization.implicit import ImplicitFactorizationModel

import numpy as np

dataset = get_movielens_dataset(variant='100K')
train, test = random_train_test_split(dataset)
print("Data loaded, ready to fit")
model = ImplicitFactorizationModel(n_iter=100,embedding_dim=50,l2=.00001,learning_rate=0.001)
print("Model set, training begins")

model.fit(train,verbose=True)

print("Model is ready, testing performance")

# rmse = rmse_score(model, test)

precision,recall = precision_recall_score(model, test)

print("precision,recall :",np.mean(precision),np.mean(recall))




from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score
from spotlight.factorization.implicit import ImplicitFactorizationModel

dataset = get_movielens_dataset(variant='20M')
train, test = random_train_test_split(dataset)
model = ImplicitFactorizationModel(n_iter=5)
model.fit(train,verbose=True)

rmse = rmse_score(model, test)



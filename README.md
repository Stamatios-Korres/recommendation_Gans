# Conditional Generative Adversarial Networks for Direct Recommendation Slate Optimization

This reposiroty implements the paper Conditional Generative Adversarial Networks for Direct Recommendation Slate Optimization. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Place the corresponding dataset in the file 'datasets/movielens/'. Our current version only support the dataset MovieLens datasets. The files must be saved in '.hdf5' format, e.g movielens_10M.hdf5 .

### Usage 

Training matrix factorization, Greedy-MLP and conditional GANs. The following scripts provide the full functionality of our methods. 

mf_spotlight.py

```


usage: mf_spotlight.py [-h] [--use_gpu [USE_GPU]]
                       [--l2_regularizer L2_REGULARIZER]
                       [--on_cluster ON_CLUSTER] 
                       [--dataset DATASET] [--experiment_name EXPERIMENT_NAME]
                       [--precision_recall PRECISION_RECALL]
                       [--map_recall MAP_RECALL] [--rmse RMSE]
                       [--mf_embedding_dim MF_EMBEDDING_DIM]
                       [--training_epochs TRAINING_EPOCHS]
                       [--batch_size BATCH_SIZE]
                       [--learning_rate LEARNING_RATE] [--optim OPTIM] [--k K]
                       [--neg_examples NEG_EXAMPLES] [--optim_gan OPTIM_GAN]
                       [-loss LOSS]
                       
```

ncf_spotlight.py

```


usage: ncf_spotlight.py [-h] [--use_gpu [USE_GPU]]
                       [--l2_regularizer L2_REGULARIZER]
                       [--on_cluster ON_CLUSTER] 
                       [--dataset DATASET] 
                       [--experiment_name EXPERIMENT_NAME]
                       [--precision_recall PRECISION_RECALL]
                       [--map_recall MAP_RECALL] 
                       [--rmse RMSE]
                       [--mlp_embedding_dim MLP_EMBEDDING_DIM]
                       [--training_epochs TRAINING_EPOCHS]
                       [--batch_size BATCH_SIZE]
                       [--learning_rate LEARNING_RATE] 
                       [--optim OPTIM] [--k K]
                       [--neg_examples NEG_EXAMPLES] 
                       [-loss LOSS]
                       
```

slate_generation.py 

```


usage: slate_generation.py [-h] 
                           [--use_gpu [USE_GPU]]
                           [--on_cluster ON_CLUSTER] 
                           [--dataset DATASET]
                           [--experiment_name EXPERIMENT_NAME]
                           [--training_epochs TRAINING_EPOCHS]
                           [--batch_size BATCH_SIZE]
                           [--learning_rate LEARNING_RATE] 
                           [--k K] 
                           [--optim_gan OPTIM_GAN]
                           [--gan_embedding_dim GAN_EMBEDDING_DIM]
                           [--gan_hidden_layer GAN_HIDDEN_LAYER] 
                           [--slate_size SLATE_SIZE]
                      
```

## Built With

* [Python 3.7](https://www.python.org/downloads/release/python-374/) - The framework used
* [Pytorch](https://pytorch.org/) - Deep Learning framework utilized 
* [numpy](https://www.numpy.org/) - Used to implement a lot of functionality 
* [scipy](https://www.scipy.org/) - Used to built sparse matrices, required for recommendation datasets

## Contributing

We used a lot of the basic training and validation utilities provided by the [spotlight](https://github.com/maciejkula/spotlight).

Utility functions were borrowed from [Edinburgh/mlpractical](https://github.com/CSTR-Edinburgh/mlpractical/tree/mlp2018-9/lab7)


## Authors

* **Timos Korres**  - [Github](https://github.com/Stamatios-Korres)




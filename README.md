# Conditional Generative Adversarial Networks for Direct Recommendation Slate Optimization

This reposiroty implements the paper Conditional Generative Adversarial Networks for Direct Recommendation Slate Optimization. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Place the corresponding dataset in the file 'datasets/movielens/'. Our current version only support the dataset MovieLens datasets. The files must be saved in '.hdf5' format, e.g movielens_10M.hdf5 .

### Usage 

What things you need to install the software and how to install them
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

```
Give examples
```

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Python 3.7](https://www.python.org/downloads/release/python-374/) - The framework used
* [Pytorch](https://pytorch.org/) - Deep Learning framework utilized 
* [numpy](https://www.numpy.org/) - Used to implement a lot of functionality 
* [scipy](https://www.scipy.org/) - Used to built sparse matrices, required for recommendation datasets

## Contributing

We used a lot of the basic training and validation utilities  provided by the * [spotlight](https://github.com/maciejkula/spotlight).

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

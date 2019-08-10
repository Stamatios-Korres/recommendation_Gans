import numpy as np 
import pandas as pd
import json 
import torch
import os
import logging
import time
import pickle

from utils.helper_functions import make_implicit
from spotlight.dataset_manilupation import delete_rows_csr, create_slates, train_test_timebased_split,random_train_test_split,train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.sampling import get_negative_samples
from spotlight.interactions import Interactions


logging.basicConfig(format='%(message)s',level=logging.INFO)


class slate_data_provider(object):

    def __init__(self, path, variant, slate_size = 3, min_movies = 0, min_viewers = 5,  movies_to_keep= -1 ):
        
        """
            Wraper class used to pre-process, store and load the corresponding dataset from the disk
        
            Parameters
            ----------
            path: string
                Path to the folder containing the dataset and where the processed data will be stored.
            variant: string
                Variant of MovieLens to be used (100K,1M,10M,20M)
            slate_size: int, optional
                Size of generated slates, i.e. number of movies each slate will consist of            
            min_movies: int, optional
               minimum number of movies a user must have seen in order to be used in the training set          
            min_viewers: int, optional 
                minimum number of viewers a movies must have seen in order to be used in the training set          
            min_viewers: int,optional
                Keep the movies_to_keep most popular movies. If movies_to_keep = -1 keep all movies
        """
        
        rel_path = path + str(slate_size)+'_slate_movielens_' + variant
        self.config = {}
        self.slate_size = slate_size


        self.min_movies = min_movies
        self.min_viewers =  min_viewers

        self.movies_to_keep = movies_to_keep
        cold_start_users = None
        if self.exists(rel_path):

            logging.info("Data exists, loading from file ... ")

            start = time.time()
            
            self.statistics = self.read_statistics(rel_path)
            
            train_slates = self.read_slates(rel_path,'_train_slates_'+str(self.movies_to_keep)+'.pkl')

            test_set  = self.load_user_vec(rel_path,'_test_set')
            valid_set = self.load_user_vec(rel_path,'_valid_set')

            test_vec_cold_start = self.load_user_vec(rel_path,'_train_cold_start')
            val_vec_cold_start = self.load_user_vec(rel_path,'_val_cold_start')

            train_vec = torch.Tensor(self.load_user_vec(rel_path,'_train_vec'))
            test_vec = torch.Tensor(self.load_user_vec(rel_path,'_test_vec'))
            valid_vec = torch.Tensor(self.load_user_vec(rel_path,'_valid_vec'))

            end = time.time()

        else:

            start = time.time()
            logging.info('Dataset is not set, creating csv files')
            from spotlight.datasets.movielens import get_movielens_dataset

            dataset, _ = get_movielens_dataset(variant=variant,
                                               path=path,
                                               min_uc=self.min_viewers, 
                                               min_sc=self.min_movies,
                                               movies_to_keep= movies_to_keep)
            self.statistics = {'num_users':dataset.num_users,
                               'num_items':dataset.num_items,
                               'interactions':dataset.__len__()}

            self.save_statistics(rel_path,self.statistics)


            ################################################
            # Create training set - slates & training_user #
            ################################################

            train_set, test_set = train_test_timebased_split(dataset, test_percentage=0.1)
            user_history = train_set
            train_set, valid_set = train_test_timebased_split(train_set, test_percentage=0.1)
            
            train_split,train_slates = create_slates(train_set,n = self.slate_size,padding_value = dataset.num_items)    
            # train_vec contains the list of movies each user has seen (users x list_movies)
            valid_rows,train_vec,_ = self.preprocess_train(train_split)

            rows_to_delete = np.delete(np.arange(dataset.num_users),valid_rows)

            # Delete examples with we have no history. 
            train_slates = np.delete(train_slates,rows_to_delete,axis=0)

            # Train use : [train_vec,train_slates]
            self.save_user_vec(rel_path,'_train_vec',train_vec.numpy())

            #########################
            # Create validation set #
            #########################
            
            #valid_history represents the user embeddings, valid_future is set to test on the examples 
            # valid_history, valid_future = train_set, valid_set
            valid_history, valid_future =  train_test_timebased_split(valid_set, test_percentage=0.2)  
            
            valid_future = valid_future.tocsr()

            val_rows,valid_vec,valid_cold_start = self.preprocess_train(valid_history.tocsr())
            val_vec_cold_start = valid_future.tocsr()[valid_cold_start,:]
            to_del = np.delete(np.arange(dataset.num_users),val_rows)
            valid_set = delete_rows_csr(valid_future.tocsr(),row_indices=list(to_del))

            # Valid use : [valid_vec,valid_set]
            self.save_user_vec(rel_path,'_valid_set',valid_set)
            self.save_user_vec(rel_path,'_valid_vec',valid_vec.numpy())

            ##################################################
            # Create test set - user_history and test slates #
            ##################################################

            # valid, test_vec,cold_start_users = self.preprocess_train(user_history.tocsr())
            # valid, test_vec,cold_start_users = self.preprocess_train(valid_history.tocsr())

            valid_history, test_set =  train_test_timebased_split(test_set, test_percentage=0.2)  
            test_vec_cold_start = test_set.tocsr()[cold_start_users,:]

            testing = np.arange(test_vec.shape[0])
            to_del = np.delete(testing,valid)

            test_set = delete_rows_csr(test_set.tocsr(),row_indices=list(to_del))

            self.save_user_vec(rel_path,'_test_set',test_set)
            self.save_user_vec(rel_path,'_train_cold_start',test_vec_cold_start)
            self.save_user_vec(rel_path,'_val_cold_start',val_vec_cold_start)
            self.save_user_vec(rel_path,'_test_vec',test_vec.numpy())
            

            # Test use : [test_vec,test_set], test_vec will be train + valid set, test_set will be the held out interactions
            
            self.create_cvs_file(rel_path, train_slates, test_set)
            end = time.time()
        
        # Get a better understanding of whether we need cold start user vectors
            
        logging.info("Took %d seconds"%(end - start))
        logging.info("{} user and {} items".format(self.statistics['num_users'],self.statistics['num_items']))

    
        self.config = {
            'train_vec': train_vec,
            'test_vec': test_vec,
            'valid_vec':valid_vec,

            'test_vec_cold_start': test_vec_cold_start,
            'val_vec_cold_start': val_vec_cold_start,
            'train_slates':train_slates,

            'test_set': test_set,
            'valid_set': valid_set,
            'num_items': self.statistics['num_items'],
            'num_user': self.statistics['num_users'],
            
        }

    def create_cvs_file(self,rel_path, train_slates,test_set):
        '''
            Function which saves the trainining slate and the test set in the disk

            Parameters
            -----------
                rel_path:string
                    Filepath to store the results
                train_slates: torch.Tensor
                    Training slates which contain the target slates
                test_set: sparse matrix in scipy.csr format
                    Contaings the held-out interactions of users to predict

        '''
        
        pd_test_set = pd.DataFrame(data={'userId':test_set.tocoo().row,  'movieId':test_set.tocoo().col})
        pd_test_set.columns = ['userId', 'movieId']
        pd_test_set.to_csv(rel_path + '_test_set_'+str(self.movies_to_keep)+'.csv', index=False)

        with open(rel_path + '_train_slates_'+str(self.movies_to_keep)+'.pkl', 'wb') as (f):
            pickle.dump(train_slates, f)

    def get_cold_start_users(self):
        '''
        Returns test cold_start_users
        '''
        return self.config['test_vec_cold_start']

    def preprocess_train(self,interactions):
        
        '''
        Turn a sparse crs format array of interactions into torch.Tensor, which will be used to index the embedding layer.

        Parameters
        ----------
            interactions: sparse matrix contatining interactions

        Output
        ---------
            valid_rows: np.array
                    containing which users in the training have indeed available training interactions
            pad_sequence: torch.Tensor (valid_users, max interactions)
                torch.Tensor for indexing the embedding layer in batches. If a user has less than max_interactions then he is padded until he reached max_interactions
                Padding_value: self.num_items. Indexes the embedding layer with zero vector
                
        '''
        row,col = interactions.nonzero()
        cold_start_users = np.arange(self.statistics['num_users'])
        valid_rows = np.unique(row)
        cold_start_users = np.delete(cold_start_users,valid_rows)
        indices = np.where(row[:-1] != row[1:])
        indices = indices[0] + 1
        vec = np.split(col,indices)
        vec = [torch.Tensor(x) for x in vec]
        return  valid_rows,torch.nn.utils.rnn.pad_sequence(vec, batch_first=True, padding_value = self.statistics['num_items']),cold_start_users
    
    def save_user_vec(self,path,filename,user_vec):
        path += filename
        path+= '_'+str(self.movies_to_keep)
        with open(path, 'wb') as (f):
            pickle.dump(user_vec, f)

    def load_user_vec(self,path,filename):
        path += filename
        path += '_'+str(self.movies_to_keep)
        with open(path, 'rb') as (f):
            return pickle.load(f)
        
    def create_interactions(self,df,num_users,num_items):
        """
        Creates a Interactions placeholder for saving user-item interactions.
        The interactions are read from a panda dataframe
        
        Input
        -----------

            df: pandas dataframe
            num_user: total number of users in the set
            num_user: total number of items in the set
        
        Output
        -------------
            Interactions class 

        """

        uid = df.userId.values
        sid = df.movieId.values
        return Interactions(uid,sid, num_users=num_users,num_items=num_items)
   
    def save_statistics(self,path,statistics):
        path = path+'_statistics_'+ str(self.movies_to_keep)+'.json'
        with open(path, 'w') as fp:
            json.dump(statistics, fp)
    
    def read_statistics(self, path):
        path = path+'_statistics_'+ str(self.movies_to_keep)+'.json'
        with open(path, 'r') as fp:
            return json.load(fp)
    
    def get_data(self):

        """
            Returns the dataset    
            
                Parameters
                ----------

                Output
                ----------
                    Tuple of: 
                        train_vec:  torch.Tensor, containing the user vec containing its training past interactions
                        train_slates:  torch.Tensor, represents the target slates the generator must predict
                        test_vec:  torch.Tensor, containing the user vec containing its test past interactions
                        valid_vec:  torch.Tensor, containing the user vec containing its validation past interactions
                        num_user: int, number of available users
                        num_items: int, number of available items
                        test_set:  Interactions class containing testing interactions
                        valid_set:  Interactions class containing validation interactions
                        val_vec_cold_start: torch.Tensors, cold start users on validation set.          
        """

        return (self.config['train_vec'], 
                self.config['train_slates'], 
                self.config['test_vec'], 
                self.config['test_set'], 
                self.config['num_user'], 
                self.config['num_items'],
                self.config['valid_vec'],
                self.config['val_vec_cold_start'],
                self.config['valid_set']     
        )


    def read_slates(self,path,filename):
        '''
            Loads the slates from the disk
        '''
        path +=filename
        with open(path, 'rb') as (f):
            return pickle.load(f)

    def exists(self, path):
        '''
            Tests whether all required files exist, returning a bool value
        '''
        print(path)
        return  os.path.exists(path + '_train_vec_'+ str(self.movies_to_keep))   and os.path.exists(path + '_test_vec_' + str(self.movies_to_keep) )  and  os.path.exists(path + '_valid_vec_' + str(self.movies_to_keep))  and os.path.exists(path + '_train_cold_start_' + str(self.movies_to_keep)) and  os.path.exists(path + '_test_set_' + str(self.movies_to_keep))


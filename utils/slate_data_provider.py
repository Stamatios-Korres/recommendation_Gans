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

    def __init__(self, path, variant, slate_size = 3, min_movies = 0, min_viewers = 5,  movies_to_keep= 1000 ):
        
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
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
            # test_set_df = pd.read_csv(rel_path + '_test_set_'+str(self.movies_to_keep)+'.csv')
            # test_set = self.create_interactions(test_set_df,self.statistics['num_users'],self.statistics['num_items'])
            # valid_set_df = pd.read_csv(rel_path + '_test_set_'+str(self.movies_to_keep)+'.csv')
            # valid_set = self.create_interactions(valid_set_df,self.statistics['num_users'],self.statistics['num_items'])
            
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

            train_set, test_set = train_test_timebased_split(dataset, test_percentage=0.2)
            train_set, valid_set = train_test_timebased_split(train_set, test_percentage=0.2)
            
            train_split,train_slates = create_slates(train_set,n = self.slate_size,padding_value = dataset.num_items)    
            train_set = None
            valid_rows,train_vec,_ = self.preprocess_train(train_split)
            rows_to_delete = np.delete(np.arange(dataset.num_users),valid_rows)
            train_slates = np.delete(train_slates,rows_to_delete,axis=0)

            #########################
            # Create validation set #
            #########################
                    
            valid_history, valid_future = train_test_timebased_split(valid_set, test_percentage=0.3)  
            
            valid_future = valid_future.tocsr()

            val_rows,valid_vec,valid_cold_start = self.preprocess_train(valid_history.tocsr())
            val_vec_cold_start = valid_future.tocsr()[valid_cold_start,:]
            to_del = np.delete(np.arange(dataset.num_users),val_rows)
            valid_set = delete_rows_csr(valid_future.tocsr(),row_indices=list(to_del))
            self.save_user_vec(rel_path,'_test_set',test_set)
            self.save_user_vec(rel_path,'_valid_set',valid_set)
            ##################################################
            # Create test set - user_history and test slates #
            ##################################################
            valid, test_vec,cold_start_users = self.preprocess_train(train_set.tocsr())
            test_vec_cold_start = test_set.tocsr()[cold_start_users,:]
            testing = np.arange(test_vec.shape[0])
            to_del = np.delete(testing,valid)
            test_set = delete_rows_csr(test_set.tocsr(),row_indices=list(to_del))
            
            self.save_user_vec(rel_path,'_train_cold_start',test_vec_cold_start)
            self.save_user_vec(rel_path,'_val_cold_start',val_vec_cold_start)

            self.save_user_vec(rel_path,'_test_vec',test_vec.numpy())
            self.save_user_vec(rel_path,'_train_vec',train_vec.numpy())
            self.save_user_vec(rel_path,'_valid_vec',valid_vec.numpy())

          
            self.create_cvs_file(rel_path, train_slates, test_set)
            end = time.time()
            
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
            # 'cold_start_users': cold_start_users,
            'num_user': self.statistics['num_users'],
            
        }

    def create_cvs_file(self,rel_path, train_slates,test_set):
        
        pd_test_set = pd.DataFrame(data={'userId':test_set.tocoo().row,  'movieId':test_set.tocoo().col})
        pd_test_set.columns = ['userId', 'movieId']
        pd_test_set.to_csv(rel_path + '_test_set_'+str(self.movies_to_keep)+'.csv', index=False)

        with open(rel_path + '_train_slates_'+str(self.movies_to_keep)+'.pkl', 'wb') as (f):
            pickle.dump(train_slates, f)

    def get_cold_start_users(self):
        
        return self.config['test_vec_cold_start']

    def preprocess_train(self,interactions):
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
        path +=filename
        with open(path, 'rb') as (f):
            return pickle.load(f)

    def exists(self, path):
        print(path)
        return  os.path.exists(path + '_train_vec_'+ str(self.movies_to_keep))   and os.path.exists(path + '_test_vec_' + str(self.movies_to_keep) )  and  os.path.exists(path + '_valid_vec_' + str(self.movies_to_keep))  and os.path.exists(path + '_train_cold_start_' + str(self.movies_to_keep)) and  os.path.exists(path + '_test_set_' + str(self.movies_to_keep))
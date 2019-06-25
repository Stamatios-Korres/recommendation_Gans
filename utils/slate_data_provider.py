import numpy as np 
import pandas as pd
import json 
import os
import logging
import time
import pickle

from utils.helper_functions import make_implicit
from spotlight.dataset_manilupation import train_test_timebased_split,random_train_test_split,train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.sampling import get_negative_samples
from spotlight.interactions import Interactions


logging.basicConfig(format='%(message)s',level=logging.INFO)


class slate_data_provider(object):

    def __init__(self, path, variant,min_movies = 5, min_viewers = 0 ):
        
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        rel_path = path + 'slate_movielens_' + variant
        self.config = {}

        self.min_movies = min_movies
        self.min_viewers =  min_viewers

        if self.exists(rel_path):
            
            start = time.time()
    
            logging.info("Data exists, loading from file ... ")

            train_df = pd.read_csv(rel_path + '_train.csv')
            test_df = pd.read_csv(rel_path + '_test.csv')

            statistics = self.read_statistics(rel_path)

            train_set = self.create_interactions(train_df,statistics['num_users'],statistics['num_items'])
            test_set = self.create_interactions(test_df,statistics['num_users'],statistics['num_items'])

            train_set = make_implicit(train_set)
            test_set = make_implicit(test_set)

            item_popularity = pd.read_csv(rel_path + '_popularity.csv',header=None).iloc[:,1]

            end = time.time()

        else:
            start = time.time()
            logging.info('Dataset is not set, creating csv files')

            dataset, item_popularity = get_movielens_dataset(variant=variant, path=path,min_uc=self.min_movies, min_sc=self.min_viewers)
            self.save_statistics(rel_path,dataset.num_users,dataset.num_items,dataset.__len__())

            train_set, test_set = train_test_timebased_split(dataset, test_percentage=0.2)

            self.create_cvs_files(rel_path, train_set, test_set,  item_popularity)
            end = time.time()
            
        logging.info("Took %d seconds"%(end - start))
        logging.info("{} user and {} items".format(train_set.num_users,train_set.num_items))

        self.config = {
            'train_set': train_set,
            'test_set': test_set,
            'item_popularity': item_popularity,
        }
    
    def get_data(self):
        return (self.config['train_set'], 
                self.config['test_set'], 
                self.config['item_popularity']
        )
    
    def save_statistics(self,path,num_users,num_items,interactions):
        statistics = { 'num_users':num_users,
                       'num_items':num_items,
                       'interactions':interactions}
        path = path+'_statistics.json'
        with open(path, 'w') as fp:
            json.dump(statistics, fp)
    
    def read_statistics(self,path):
        path = path+'_statistics.json'
        with open(path, 'r') as fp:
            return json.load(fp)
        
    def create_interactions(self,df,num_users,num_items):
        
        """
        Creates a Interactions placeholder for saving user-item interactions.
        The interactions are read from a panda dataframe

        
        Parameters
        -----------

            df: pandas dataframe
            num_user: total number of users in the set
            num_user: total number of items in the set
        
        Returns:
            Interactions class 

        """
        uid = df.userId.values
        sid = df.movieId.values
        timestamps = df.timestamp.values
        ratings = df.rating.values
        return Interactions(uid,sid,ratings,timestamps,num_users=num_users,num_items=num_items)

    def create_cvs_files(self, rel_path, train, test, item_popularity):
        
        pd_train = pd.DataFrame(data={'userId':train.user_ids,  'movieId':train.item_ids,  'rating':train.ratings,'timestamp':train.timestamps})
        pd_train.columns = ['userId', 'movieId', 'rating','timestamp']

        pd_test = pd.DataFrame(data={'userId':test.user_ids,  'movieId':test.item_ids,  'rating':test.ratings,'timestamp':test.timestamps})
        pd_test.columns = ['userId', 'movieId', 'rating','timestamp']

        pd_train.to_csv(rel_path + '_train.csv', index=False)
        pd_test.to_csv(rel_path + '_test.csv', index=False)

        item_popularity.to_csv(rel_path + '_popularity.csv', header=False)

    def exists(self, path):
        return os.path.exists(path + '_train.csv') and os.path.exists(path + '_popularity.csv') and os.path.exists(path + '_test.csv') 
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


class data_provider(object):

    def __init__(self, path, variant, negative_per_positive, movies_to_keep = -1):

        """
        Wraper class used to pre-process, store and load the corresponding dataset from the disk
        
            Parameters
            ----------
            path: string
                Path to the folder containing the dataset and where the processed data will be stored.
            variant: string
                Variant of MovieLens to be used (100K,1M,10M,20M)
            negative_per_positive: int
                How many negative examples to samples for each positive interaactions           
            movies_to_keep: int
                Keep the movies_to_keep most popular movies. If movies_to_keep = -1 keep all movies
            
        """
        self.movies_to_keep = movies_to_keep             
        rel_path = path + 'movielens_' + variant
        self.config = {}
        
        # Check if the specific pre-processing has already taken place. If yes, load the file from the disk
        if self.exists(rel_path):
            
            start = time.time()
    
            logging.info("Data exists, loading from file ... ")
            train_df = pd.read_csv(rel_path + '_train_'+str(self.movies_to_keep)+'.csv')
            valid_df = pd.read_csv(rel_path + '_valid_'+str(self.movies_to_keep)+'.csv')
            test_df = pd.read_csv(rel_path + '_test_'+str(self.movies_to_keep)+'.csv')

            statistics = self.read_statistics(rel_path)

            train_set = self.create_interactions(train_df,statistics['num_users'],statistics['num_items'])
            valid_set = self.create_interactions(valid_df,statistics['num_users'],statistics['num_items'])
            test_set = self.create_interactions(test_df,statistics['num_users'],statistics['num_items'])

            train_set = make_implicit(train_set)
            valid_set = make_implicit(valid_set)
            test_set = make_implicit(test_set)

            item_popularity = pd.read_csv(rel_path + '_popularity_'+str(self.movies_to_keep)+'.csv',header=None).iloc[:,1]

            neg_examples = self.read_negative_examples(rel_path + '_ngt_'+str(self.movies_to_keep)+'.pkl')
            
            end = time.time()
        # The specific configuration has not been done in the past. Split data and create the negative examples. 
        else:
            start = time.time()
            logging.info('Dataset is not set, creating csv files')

            dataset, item_popularity = get_movielens_dataset(variant=variant, path=path, movies_to_keep = movies_to_keep)
            
            self.save_statistics(rel_path,dataset.num_users,dataset.num_items,dataset.__len__())

            dataset = make_implicit(dataset)
            train_set, test_set = train_test_timebased_split(dataset, test_percentage=0.1)
            train_set, valid_set = train_test_timebased_split(train_set, test_percentage=0.1)
            
            
            neg_examples = get_negative_samples(dataset, train_set.__len__())
            self.create_cvs_files(rel_path, train_set, valid_set, test_set, neg_examples, item_popularity)
            end = time.time()
            
        logging.info("Took %d seconds"%(end - start))
        logging.info("{} user and {} items".format(train_set.num_users,train_set.num_items))
        
        # Disctionary which contains the resulting processed data
        self.config = {
            'train_set': train_set,
            'valid_set': valid_set,
            'test_set': test_set,
            'item_popularity': item_popularity,
            'neg_examples':neg_examples
        }
    
    def get_timebased_data(self):
        """
            Returns the dataset    
            
                Parameters
                ----------

                Output
                ----------
                    Tuple of: 
                        train_set:  Interactions class containing training interactions
                        test_set:  Interactions class containing testing interactions
                        valid_set:  Interactions class containing validation interactions
                        neg_examples: list of tuples (user_id,movie_id) 
                        item_popularity: pandas.dataframe containing the popularity of each movie
            
        """
        return (self.config['train_set'], 
                self.config['valid_set'],
                self.config['test_set'], 
                self.config['neg_examples'],
                self.config['item_popularity']
        )
    
    def save_statistics(self,path,num_users,num_items,interactions):
        statistics = { 'num_users':num_users,
                       'num_items':num_items,
                       'interactions':interactions}
        path = path+'_statistics_'+str(self.movies_to_keep)+'.json'
        with open(path, 'w') as fp:
            json.dump(statistics, fp)
    
    def read_statistics(self,path):
        path = path+'_statistics_'+str(self.movies_to_keep)+'.json'
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

    def create_cvs_files(self, rel_path, train, valid, test, neg_examples, item_popularity):
        p = pickle.Pickler(open(rel_path + '_ngt_'+str(self.movies_to_keep)+'.pkl', 'wb')) 
        p.fast = True 
        p.dump(neg_examples) 
        pd_train = pd.DataFrame(data={'userId':train.user_ids,  'movieId':train.item_ids,  'rating':train.ratings,'timestamp':train.timestamps})
        pd_train.columns = ['userId', 'movieId', 'rating','timestamp']
        pd_valid = pd.DataFrame(data={'userId':valid.user_ids,  'movieId':valid.item_ids,  'rating':valid.ratings,'timestamp':valid.timestamps})
        pd_valid.columns = ['userId', 'movieId', 'rating','timestamp']
        pd_test = pd.DataFrame(data={'userId':test.user_ids,  'movieId':test.item_ids,  'rating':test.ratings,'timestamp':test.timestamps})
        pd_test.columns = ['userId', 'movieId', 'rating','timestamp']

        pd_train.to_csv(rel_path + '_train_'+str(self.movies_to_keep)+'.csv', index=False)
        pd_valid.to_csv(rel_path + '_valid_'+str(self.movies_to_keep)+'.csv', index=False)
        pd_test.to_csv(rel_path + '_test_'+str(self.movies_to_keep)+'.csv', index=False)
        item_popularity.to_csv(rel_path + '_popularity_'+str(self.movies_to_keep)+'.csv', header=False)

    def read_negative_examples(self, target):
        with open(target, 'rb') as (f):
            return pickle.load(f)

    def exists(self, path):
        return os.path.exists(path + '_train_'+str(self.movies_to_keep)+'.csv') and os.path.exists(path + '_popularity_'+str(self.movies_to_keep)+'.csv') and os.path.exists(path + '_valid_'+str(self.movies_to_keep)+'.csv') and os.path.exists(path + '_test_'+str(self.movies_to_keep)+'.csv') and os.path.exists(path + '_ngt_'+str(self.movies_to_keep)+'.pkl')

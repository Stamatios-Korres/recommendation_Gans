
import torch
import numpy as np
from torch.utils import data
import spotlight.interactions 
import os.path
import pandas as pd
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.sampling import get_negative_samples
import pickle
from spotlight.cross_validation import train_test_timebased_split


class data_provider(object):
    
    def __init__(self,path, variant,negative_per_positive):

        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        rel_path = path+'movielens_'+variant

        if self.exists(rel_path):
            train_set = pd.read(rel_path+'_train.csv')
            valid_set = pd.read(rel_path+'_valid.csv')
            test_set  = pd.read(rel_path+'_test.csv')
            neg_examples = self.read_negative_examples(rel_path+'_ngt.pkl')
        else:
            dataset, item_popularity = get_movielens_dataset(variant=variant,path=path)
            train_set,test_set = train_test_timebased_split(dataset,test_percentage=0.2)
            train_set,valid_set = train_test_timebased_split(train_set,test_percentage=0.2)
            neg_examples = get_negative_samples(dataset,(dataset.__len__())*negative_per_positive)    
            self.create_cvs_files(rel_path,train_set,valid_set,test_set,neg_examples)
        return train_set,valid_set,test_set,neg_examples,item_popularity
  
    def create_cvs_files(self,rel_path,train,valid,test,neg_examples):
        with open(rel_path+'_ngt.pkl') as f:
            pickle.dump(neg_examples,f)

        pd_train = pd.DataFrame(data={'userId':train.user_ids,'movieId':train.item_ids,'rating':train.ratings},colunns =['userId','movieId','rating'])
        pd_valid = pd.DataFrame(data={'userId':valid.user_ids,'movieId':valid.item_ids,'rating':valid.ratings},colunns =['userId','movieId','rating'])
        pd_test = pd.DataFrame(data={'userId':test.user_ids,'movieId':test.item_ids,'rating':test.ratings},colunns =['userId','movieId','rating'])
     
        pd_train.to_csv(os.path.join(rel_path,'_train.csv'),index=False)
        pd_valid.to_csv(os.path.join(rel_path,'_valid.csv'),index=False)
        pd_test.to_csv(os.path.join(rel_path,'_test.csv'),index=False)

        

    def read_negative_examples(self,target):
        with open(target,'rb') as f: 
            return pickle.load(f) 

    def exists(self,path):

        return (    os.path.exists(path+'_train.csv')
                and os.path.exists(path+'_valid.csv')
                and os.path.exists(path+'_test.csv')
                and os.path.exists(path+'_ngt.pkl')
        )


    
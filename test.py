import numpy as np 
import pandas as pd 
import mf
from nfc.src.mlp import MLP as mlp


f = open('MSD/kaggle_visible_evaluation_triplets.txt', 'r')
song_to_count = dict()
for line in f:
    _,song,_ = line.strip().split('\t')
    if song in song_to_count:
        song_to_count[song]+=1
    else:
        song_to_count[song] = 1

f.close()
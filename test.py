import numpy as np 
import pandas as pd 
import mf
from nfc.src.mlp import MLP as mlp

f = open('MSD/train_triplets.txt', 'r')
song_to_count = dict()
songs_per_user = dict()
user_to_songs = dict() 
for line in f:
    user,song,_ = line.strip().split('\t')
    if song in song_to_count:
        song_to_count[song]+=1
    else:
        song_to_count[song] = 1
    if user in user_to_songs:
        user_to_songs[user].add(song)
    else:
        user_to_songs[user] = set([song])
    if user in songs_per_user:
        songs_per_user[user]+=1
    else:
        songs_per_user[user] = 1
f.close()

# Data preprocessing - Only keep users with more than 20 songs and songs that have been listened by at least 200 users!
valid_songs = set()
user_to_songs = {k: v for k, v in user_to_songs.items() if songs_per_user[k]>20}
for user,user_songs in user_to_songs.items():
    new_list = [s for s in list(user_songs) if song_to_count[s] >= 200]
    user_to_songs[user] = new_list
    for valid_song in new_list:
        valid_songs.add(valid_song)
user_to_songs = {k: v for k, v in user_to_songs.items() if v}

# Creating sparse matrix Coordinate Format (COO)

users = {}
songs = {}
i = 0 # Index for users
for user in user_to_songs.keys():
    users[i] = user
    i+=1
j = 0 # index for songs
for song in valid_songs:
    songs[j] = song
    j+=1
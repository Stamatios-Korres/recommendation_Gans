import numpy as np 
import pandas as pd 
import mf
from scipy.sparse import csr_matrix
from mf import MF 
from mf_pytorch_train import train_test_split,implicit

f = open('ml-1m/ratings.dat', 'r')
songs_count = dict()
songs_per_user = dict()
user_to_songs = dict() 
valid_songs = set()
for line in f:
    user,song,_,_ = line.strip().split('::')
    valid_songs.add(song)
    if song in songs_count:
        songs_count[song]+=1
    else:
        songs_count[song] = 1
    if user in user_to_songs:
        user_to_songs[user].add(song)
    else:
        user_to_songs[user] = set([song])
    if user in songs_per_user:
        songs_per_user[user]+=1
    else:
        songs_per_user[user] = 1
f.close()

min_number_of_songs = 20 
min_number_of_listeners = 0

print("File was read!")

# valid_songs = [s for s in valid_songs if songs_count[s] >= min_number_of_listeners]
user_to_songs = {k: v for k, v in user_to_songs.items() if songs_per_user[k]>=min_number_of_songs}

print("DataSet was set. We have",len(valid_songs),"songs")

# Data preprocessing - Only keep users with more than 20 songs and songs that have been listened by at least 200 users!

# for user,user_songs in user_to_songs.items():
#     new_list = [s for s in list(user_songs) if songs_count[s] >= min_number_of_listeners]
#     user_to_songs[user] = new_list

# user_data = {k: v for k, v in user_to_songs.items() if v}
# user_to_songs = None # Release Memory 

# print("Number of users are:",len(user_data))
# Creating sparse matrix Coordinate Format (COO)

users = {}
songs = {}
i = 0 # Index for users
for user in user_to_songs.keys():
    users[user] = i
    i+=1
j = 0 # index for songs
for song in valid_songs:
    songs[song] = j
    j+=1
    
number_of_users = len(users)
number_of_songs = len(songs)
print(number_of_users,number_of_songs)

row = []
column = []
data = []
for user,user_index in users.items():
    for song,song_index in songs.items():
        row.append(user_index)
        column.append(song_index)
        # Implicit feedback (1 if the user has interacted with the song, else 0 )
        data.append(1 if (song in user_to_songs[user]) else 0)

# user_data = None # Release Memory  


print("Preprocess is done, starting training")

row = np.array(row)
column = np.array(column)
data = np.array(data)
R = csr_matrix((data, (row, column)),shape=(number_of_users,number_of_songs))
print("Sparse matrix created")
train_test_split(R)

# mf = MF(R, 5, 0.005, 0.01, 20)
# mf.train()

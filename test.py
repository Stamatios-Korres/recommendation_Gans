import numpy as np 
import pandas as pd 
import mf
from scipy.sparse import csr_matrix
from mf import MF 
from mf_pytorch_train import *

# Read data from file and store them to data_structures
# df = pd.read_csv('output_list.txt', delimiter="\t", header=None, names=["a", "b", "c"])

f = open('MSD/train_triplets.txt', 'r')
songs_count = dict()
songs_per_user = dict()
user_to_songs = dict() 
valid_songs = set()

for line in f:
    user,song,_ = line.strip().split('\t')
    valid_songs.add(song)
    if song in songs_count:
        songs_count[song] += 1
    else:
        songs_count[song] = 1
    if user in user_to_songs:
        user_to_songs[user].add(song)
    else:
        user_to_songs[user] = set([song])
    if user in songs_per_user:
        songs_per_user[user] += 1
    else:
        songs_per_user[user] = 1
f.close()

min_number_of_songs = 20 
min_number_of_listeners = 200

print("File was read!")

valid_songs = [s for s in valid_songs if songs_count[s] >= min_number_of_listeners]
user_to_songs = {k: v for k, v in user_to_songs.items() if songs_per_user[k]>=min_number_of_songs}

# print("DataSet was set. We have",len(valid_songs),"songs")

# Data preprocessing - Only keep users with more than 20 songs and songs that have been listened by at least 200 users!

interactions = 0 
for user,user_songs in user_to_songs.items():
    new_list = [s for s in list(user_songs) if songs_count[s] >= min_number_of_listeners]
    user_to_songs[user] = new_list
    interactions += len(new_list)

user_data = {k: v for k, v in user_to_songs.items() if v}
user_to_songs = None # Release Memory 

# print("Number of users are:",len(user_data))
print("Number of interactions are:",interactions)

# Creating sparse matrix Coordinate Format (COO)

users = {}
songs = {}
i = 0 # Index for users
for user in user_data.keys():
    users[i] = user
    i+=1
j = 0 # index for songs
for song in valid_songs:
    songs[j] = song
    j+=1
    
number_of_users = len(users)
number_of_songs = len(songs)
print(number_of_users,number_of_songs)

# row = []
# column = []
# data = []
# for user_index,user in users.items():
#     for song_index,song in songs.items():
#         row.append(user_index)
#         column.append(song_index)
#         # Implicit feedback (1 if the user has interacted with the song, else 0 )
#         data.append(1 if (song in user_data[user]) else 0)

# user_data = None # Release Memory  


# print("Preprocess is done, starting training")

# row = np.array(row)
# column = np.array(column)
# data = np.array(data)
# R = csr_matrix((data, (row, column)),shape=(number_of_users,number_of_songs))
# train_test_split(R)
# # mf = MF(R, 5, 0.05, 1, 20)

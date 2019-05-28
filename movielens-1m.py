import numpy as np 
# import pandas as pd 
# import mf
# from nfc.src.mlp import MLP as mlp

f = open('ml-1m/ratings.dat', 'r')
# movies_to_count = dict()
movies_per_user = dict()
user_to_movies = dict() 
interactions_slow_way = 0 
for line in f:
    user,movie,_,_ = line.strip().split('::')
    interactions_slow_way+=1
    # if movie in movies_to_count:
    #     movies_to_count[movie]+=1
    # else:
    #     movies_to_count[movie]=1
    if user in user_to_movies:
        user_to_movies[user].add(movie)
    else:
        user_to_movies[user]=set([movie])
    if user in movies_per_user:
        movies_per_user[user]+=1
    else:
        movies_per_user[user]=1
f.close()
# print("Number of valid users", len(user_to_movies))

# Data preprocessing - Only keep users with more than 20 songs and songs that have been listened by at least 200 users!
valid_movies = set()
interactions = 0 
valid_users = {k:v for k,v in user_to_movies.items() if movies_per_user[k]>=20}
for user,user_movies in valid_users.items():
    interactions+=len(user_movies)
    for valid_movie in user_movies:
        valid_movies.add(valid_movie)
user_data = {k: v for k, v in valid_users.items() if v}

 
print("Number of users are:",len(user_data))
print("Number of movies are:",len(valid_movies))
print("Number of interactions are:",interactions)
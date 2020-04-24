"""Model training module"""

from sqlalchemy import create_engine
import pandas as pd
import os
import numpy as np
from sklearn.decomposition import NMF
from joblib import dump


#Connect to the server
HOST = 'localhost'
PORT = '5432'
DB = 'movies'

conn_string_mac = f'postgres://{HOST}:{PORT}/{DB}'
engine = create_engine(conn_string_mac)

#Create engine
engine = create_engine(conn_string_mac)

#Create dfs from database
df_links = pd.read_sql_query('SELECT * from links',con=engine)
df_tags = pd.read_sql_query('SELECT * from tags',con=engine)
df_ratings = pd.read_sql_query('SELECT * from ratings',con=engine)
df_movies = pd.read_sql_query('SELECT * from movies',con=engine)
df_movies['title_new'] = df_movies['title'].replace('[^a-z]*$', '', regex = True)
df_movies['title_new'] = df_movies['title'].replace('\(.*', '', regex = True)
df_movies['title_new'] = df_movies['title_new'].str.lower()

#Movie Dictionary
movie_id_dict = dict(zip(df_movies['movieId'], df_movies['title_new']))

def mean_rating():
    mean_rating = df_ratings['rating'].mean()
    return mean_rating

def create_sparse(ratings):
    user_movie_matrix = ratings.pivot(index = 'userId', columns = 'movieId',
                                    values = 'rating')
    return user_movie_matrix

def create_dense():
    user_sparse = create_sparse(df_ratings)
    mean_rating = df_ratings['rating'].mean()
    user_movie_matrix = user_sparse.fillna(value = mean_rating)
    return user_movie_matrix

def model_fit(dense_matrix):
    model = NMF(n_components=150, init='random', random_state=10)
    model.fit(dense_matrix)
    return model

def dump_model(model):
    dump(model, './NMF.joblib')

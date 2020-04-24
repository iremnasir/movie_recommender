from model import create_dense, movie_id_dict, mean_rating
import numpy as np
from scipy.spatial import distance
from recommender import convert_user_input
import pandas as pd
from fuzzywuzzy import fuzz


def cosine_similarity(number_of_recomm, user_input_movies, user_input_ratings):
    #Create the mean-filled dense_matrix
    dense_matrix = create_dense()
    #Create the user array
    mean_rating_1 = mean_rating()
    user = np.repeat(mean_rating_1, dense_matrix.shape[1])
    user_df = pd.DataFrame([user], columns = dense_matrix.columns)
    #Collect user input
    user_mov_index = convert_user_input(user_input_movies, user_input_ratings)
    #Impute user ratings
    for mov_id in user_mov_index:
        user_df[mov_id[0]] = mov_id[1]
    #Append it to the original user_movie_matrix
    dense_matrix_user = pd.concat([dense_matrix, user_df], ignore_index=False)
    #Create user-user sparse matrix
    UU = np.zeros((len(dense_matrix_user), len(dense_matrix_user)))
    UU = pd.DataFrame(UU, index=dense_matrix_user.index, columns=dense_matrix_user.index)
    # calculate pairwise similarities
    for u in UU.index:
        for v in UU.columns:
        # 2. step: calculate similarities
            UU.loc[u, v] = 1-distance.correlation(dense_matrix_user.loc[u],
                                                    dense_matrix_user.loc[v])
    active_user = 0
    # find similarities for active_user and sort it, take 1 to 5 entries
    # entry at 0 contains the similrity with itself
    neighbors = UU.loc[active_user].sort_values(ascending=False)[1:6]
    #Final matrix
    neighbors_m = dense_matrix_user.loc[neighbors.index]
    #Take the first user and suggest movies that person liked
    random_mov = np.random.randint(6)
    movies_list = list(neighbors_m.iloc[random_mov].sort_values(ascending = False)
                        .head(number_of_recomm*2).index.map(movie_id_dict))

    return movies_list

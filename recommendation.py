import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
# load the required datasets
def get_movie_recommendation(user_id, title1, rating1, title2, rating2, title3, rating3,genre1,genre2,genre3):
    movie_data = pd.read_csv('data/movies_metadata.csv')
    ratings = pd.read_csv('data/ratings_small.csv')
    # drop unwanted columns
    movie_data = movie_data.drop(['belongs_to_collection', 'homepage', 'tagline'], axis=1)
    movie_data = movie_data.dropna()
    
    movie_data = movie_data.rename({'id': 'movieId'}, axis=1)
    movie_data['movieId'] = movie_data['movieId'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)

    # Merge movie metadata with ratings based on 'movieId'
    movie_ratings = pd.merge(ratings, movie_data, on='movieId')
    movie_ratings = movie_ratings.head(10000)

    # creare a dictionary to map movie titles to id of movies
    movie_title_to_id = {
        'Movie A': 6,
        'Movie B': 7,
        'Movie C': 8,
        
    }

    # new dataframe for new user
    new_user_id = user_id
    new_user_ratings = {'title': [title1, title2, title3],
                        'rating': [rating1, rating2, rating3],
                        'genres': [genre1, genre2, genre3]}

    new_user_df = pd.DataFrame(new_user_ratings)
    new_user_df['movieId'] = new_user_df['title'].map(movie_title_to_id) 

    # Dropping title column
    new_user_df = new_user_df.drop(columns=['title'])

    # adding new user dataframe to existing dataframe
    new_user_df['userId'] = new_user_id
    movie_ratings = pd.concat([movie_ratings, new_user_df], ignore_index=True)

    # user-movie matrix based on ratings
    rating_matrix = movie_ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

    # user-movie matrix based on genres
    genres_matrix = movie_ratings.groupby(['userId', 'movieId'])['genres'].apply(lambda x: ' '.join(x)).unstack().fillna('')

    # Replace ratings with -1 for less than 3, 1 for 3 or more, and 0 for unrated movies
    rating_matrix = rating_matrix.applymap(lambda x: -1 if x < 3 else 1 if x >= 3 else 0)

    # empty string handling
    genres_matrix = genres_matrix.replace('', np.nan)

    genres_matrix = genres_matrix.applymap(lambda x: 0 if pd.isna(x) else 1)

    # combining genre and rating matrix
    user_movie_matrix = pd.concat([rating_matrix, genres_matrix], axis=1)

    # Calculate cosine similarity between the new user and all existing users based on genres
    user_genre_sim = cosine_similarity(user_movie_matrix)

    # Sort users by cosine similarity in descending order 
    similar_users_genre = user_genre_sim[new_user_id].argsort()[::-1][1:101]

    # taking user IDs for the top 100 similar users
    similar_100_users = user_movie_matrix.iloc[similar_users_genre].index

    # Based on ratings, determine the cosine similarity between the new user and the top 100 similar users.
    user_rating_sim = cosine_similarity(user_movie_matrix.loc[similar_100_users])



    # Obtain movies that the new user hasn't seen and that the top 100 comparable users have positively reviewed.
    postive_movie_rating = rating_matrix.loc[similar_100_users] == 1
    recommended_movie_rating = postive_movie_rating.any(axis=0)
    recommended_movie_rating = recommended_movie_rating[recommended_movie_rating].index

    recommended_movies_rating = movie_ratings[movie_ratings['movieId'].isin(recommended_movie_rating)]['title']

    recommended_movie_names_set = set(recommended_movies_rating)

    # final movie recommendation
    filtered_recommended_movies = random.sample(list(recommended_movie_names_set), min(10, len(recommended_movie_names_set)))



    return filtered_recommended_movies



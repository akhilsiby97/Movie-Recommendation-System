# movie_recommendation.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten
from tensorflow.keras.optimizers import Adam

def movie_recommendation_model(df):
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    df['user'] = user_encoder.fit_transform(df['userId'])
    df['movie'] = movie_encoder.fit_transform(df['movieId'])

    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)

    num_users = df['user'].nunique()
    num_movies = df['movie'].nunique()

    user_input = Input(shape=(1,), name='user_input')
    movie_input = Input(shape=(1,), name='movie_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=50, input_length=1)(user_input)
    movie_embedding = Embedding(input_dim=num_movies, output_dim=50, input_length=1)(movie_input)

    dot_product = Dot(axes=1)([user_embedding, movie_embedding])
    flat = Flatten()(dot_product)

    model = Model(inputs=[user_input, movie_input], outputs=flat)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model



def search_movie_recommendations(user_id, model, df, movie_name, num_movies_to_recommend=10):
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    df['user'] = user_encoder.fit_transform(df['userId'])
    df['movie'] = movie_encoder.fit_transform(df['movieId'])

    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
    all_movie_ids = df['movie'].unique()
    all_movie_titles = df[['movie', 'title']].drop_duplicates()

    if movie_name:
        # Verify if the movie's name is present in the dataset.
        existing_movie = df[df['title'] == movie_name]
        if existing_movie.empty:
            new_movie_id = df['movie'].max() + 1
            new_rating = np.random.uniform(0, 5)
            new_row = {'userId': user_id, 'movieId': f'new_movie_{new_movie_id}', 'rating': new_rating, 'title': movie_name}
            df = df.append(new_row, ignore_index=True)

            df['user'] = user_encoder.transform(df['userId'])

            # Verify whether 'new_movie_' is present in the movieId column.
            x = ~df['movieId'].astype(str).str.contains('new_movie_') & df['movieId'].notna() & df['movieId'].astype(str).str.isdigit()
            df.loc[x, 'movie'] = movie_encoder.transform(df.loc[x, 'movieId'])
                
            # train the model with the updated dataset
            X_train_user = train_df['user'].values
            X_train_movie = train_df['movie'].values
            y_train = train_df['rating'].values

            model.fit([X_train_user, X_train_movie], y_train,
                      epochs=10, batch_size=64)

    # movies that is not rated by the user
    movies_not_rated = np.setdiff1d(all_movie_ids, df[df['user'] == user_id]['movie'].values)

   # Generate input data for the user and unrated films.
    user_input_data = np.full_like(movies_not_rated, user_id)
    movie_input_data = movies_not_rated

    # predict ratings for films that users haven't left.
    predicted_ratings = model.predict([user_input_data, movie_input_data])


    recommendations = list(zip(movies_not_rated, predicted_ratings.flatten()))

    # Arrange the recommendations in descending order of expected rating
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Get the top N recommended movie IDs
    top_recommendations = [movie_id for movie_id, _ in recommendations[:num_movies_to_recommend]]

    top_movie_titles = all_movie_titles[all_movie_titles['movie'].isin(top_recommendations)]['title']

    return top_movie_titles.tolist()
from google.cloud import datastore
import datetime
from datetime import datetime
import random
from flask import Flask, render_template, redirect
import google.oauth2.id_token
from flask import Flask, render_template, request
from google.auth.transport import requests
from recommendation import get_movie_recommendation
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from search import search_movie_recommendations
from search import movie_recommendation_model
app = Flask(__name__)

datastore_client = datastore.Client()
firebase_request_adapter = requests.Request()

movie_data = pd.read_csv('data/movies_metadata.csv')
ratings = pd.read_csv('data/ratings_small.csv')
movie_data = movie_data.drop(['belongs_to_collection','homepage','tagline'], axis=1)
movie_data = movie_data.dropna()
movie_data = movie_data.rename({'id':'movieId'}, axis=1)
movie_data['movieId'] = movie_data['movieId'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)
# Merge movie metadata with ratings based on 'movieId'
movie_ratings = pd.merge(ratings, movie_data, on='movieId')
df = movie_ratings.head(5000)


def createUser(claims):
    entity_key = datastore_client.key('UserStore', claims['email'])
    entity = datastore.Entity(key = entity_key)
    entity.update({
        'user_name': claims['email'],

    })
    datastore_client.put(entity)

def retrieveUser(claims):
    entity_key = datastore_client.key('UserStore', claims['email'])
    entity = datastore_client.get(entity_key)
    return entity
  
def  user_choice_details(email, user_id,movie_ratings): 
    ids = []
    
    for movie_rating in movie_ratings:
        movie = movie_rating.get('movie')
        rating = movie_rating.get('rating')
        genre = movie_rating.get('genre')
        
        entity_key = datastore_client.key('movie', movie)  
        entity = datastore.Entity(key=entity_key)
        entity.update({
            'user_id': user_id,
            'movie': movie,
            'rating': rating,
            'genre': genre,
            'email': email,
            'date': datetime.now(),
        })
        datastore_client.put(entity)
        ids.append(entity.key.id)
    
    return ids
  
def fetch_movies(email):
    query = datastore_client.query(kind='movie')
    query.add_filter('email', '=', email)
    query.order = ['-date']
    results = list(query.fetch(limit=3))
    return results
    
@app.route('/')
def root():
  id_token = request.cookies.get("token")
  print(id_token)
  error_message = None
  claims = None
  recommended_movies=None
  if id_token:
    try:
        claims = google.oauth2.id_token.verify_firebase_token(id_token,
        firebase_request_adapter)
        user_info = retrieveUser(claims)
        print('***',user_info)
        if user_info == None:
            createUser(claims)
            return render_template('newuser.html')
        email = claims['email']
        results = fetch_movies(email)
        user_id = results[0]['user_id']
        title1 = results[0]['movie']
        rating1 = results[0]['rating']
        genre1 = results[0]['genre']
        title2 = results[1]['movie']
        rating2 = results[1]['rating']
        genre2 = results[1]['genre']
        title3 = results[2]['movie']
        rating3 = results[2]['rating']
        genre3 = results[2]['genre']


  # Call the recommendation function from recommendation.py
        recommended_movies = get_movie_recommendation(user_id, title1,rating1,title2,rating2,title3,rating3,genre1,genre2,genre3)
    except ValueError as exc:
        error_message = str(exc)
            
    # Get user input from the frontend
  
  return render_template('index.html', recommended_movies=recommended_movies)

@app.route('/rate_movie',methods=['POST'])
def rate_movie():
 return render_template('rate_movie.html')

@app.route('/new_user',methods=['POST'])
def new_user():
   id_token = request.cookies.get("token")
   error_message = None
   claims = None
   claims = google.oauth2.id_token.verify_firebase_token(id_token,
  firebase_request_adapter)
   user_id = random.randint(500, 600)
   title1 = request.form['movie']
   rating1 = int(request.form['rating'])
   genre1 = request.form['genre1']
   title2 = request.form['movie2']
   rating2 = int(request.form['rating2'])
   genre2 = request.form['genre2']
   title3 = request.form['movie3']
   rating3 = int(request.form['rating3'])
   genre3 = request.form['genre3']
   movie_ratings = [{'movie': title1, 'rating': rating1,'genre':genre1},
                {'movie': title2, 'rating': rating2, 'genre':genre2},
                {'movie': title3, 'rating': rating3, 'genre':genre3}]
   movie_base = user_choice_details(claims['email'],user_id, movie_ratings)
   

  # Call the recommendation function from recommendation.py
   recommended_movies = get_movie_recommendation(user_id, title1,rating1,title2,rating2,title3,rating3,genre1,genre2,genre3)
   return render_template('index.html', recommended_movies=recommended_movies)

@app.route('/movie_rate',methods=['POST'])
def movie_rate():
  id_token = request.cookies.get("token")
  error_message = None
  claims = None
  claims = google.oauth2.id_token.verify_firebase_token(id_token,
  firebase_request_adapter)
  results = fetch_movies(claims['email'])
  latest_movie = results[0]
  user_id = latest_movie['user_id']
  title1 = request.form['movie']
  rating1 = request.form['rating']
  genre1=request.form['genre']
  movie_ratings = [{'movie': title1, 'rating': rating1,'genre':genre1}]
  movie_base = user_choice_details(claims['email'],user_id, movie_ratings)
  return render_template('rate_movie.html')

@app.route('/home', methods=['POST'])
def home():
    return redirect('/')


@app.route('/movie_search', methods=['POST'])
def searchmovie():
    id_token = request.cookies.get("token")
    error_message = None
    claims = None
    claims = google.oauth2.id_token.verify_firebase_token(id_token,
    firebase_request_adapter)
    results = fetch_movies(claims['email'])

    user_id = results[0]['user_id']
    title = request.form['Search_movie']
   
    model = movie_recommendation_model(df)
    recommended_movies = search_movie_recommendations(user_id, model, df, movie_name=title, num_movies_to_recommend=10)
    
    return render_template('index.html', recommended_movies=recommended_movies)


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8080, debug=True)

import streamlit as st
import pandas as pd
import pickle
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load the movie list and similarity matrix
movies_list = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies = pd.DataFrame(movies_list)

# Function to create a session with retry mechanism
def create_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 504)
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Function to fetch the movie poster
def fetch_poster(movie_id):
    session = create_session()
    try:
        response = session.get(
            f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=caed4707ae9fa9534db211224686bfa1&language=en-US',
            timeout=20
        )
        response.raise_for_status()  # Check if the request was successful
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/original{poster_path}"
        else:
            st.warning(f"No poster found for movie ID: {movie_id}")
            return None
    except requests.exceptions.Timeout:
        st.error("The request timed out. Please try again later.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching poster: {e}")
        return None

# Function to recommend movies
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movie_names = []
    recommended_movie_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        poster = fetch_poster(movie_id)
        if poster:
            recommended_movie_posters.append(poster)
            recommended_movie_names.append(movies.iloc[i[0]].title)
    return recommended_movie_names, recommended_movie_posters

# Streamlit UI
st.title('Movies Recommender System')
movie_list = movies['title'].values
movie_name = st.selectbox("What are you looking for today", movie_list)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(movie_name)
    if recommended_movie_names and recommended_movie_posters:
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, col in enumerate([col1, col2, col3, col4, col5]):
            with col:
                st.text(recommended_movie_names[i])
                st.image(recommended_movie_posters[i])


import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from justwatch import JustWatch

TMDB_API_KEY = "5a803f3304d624203118b6f8a661c896"

def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

def build_similarity_matrices(movies, ratings):
    merged = pd.merge(ratings, movies, on='movieId')
    user_movie_matrix = merged.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    collab_sim = cosine_similarity(user_movie_matrix.T)
    movie_index = pd.Series(user_movie_matrix.columns)

    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))
    content_sim = cosine_similarity(tfidf_matrix)
    content_index = pd.Series(movies.index, index=movies['title'])
    return collab_sim, content_sim, movie_index, content_index

def hybrid_recommend(movie_title, movies, collab_sim, content_sim, movie_index, content_index, top_n=5):
    if movie_title not in movie_index.values or movie_title not in content_index:
        return pd.DataFrame()

    collab_idx = movie_index[movie_index == movie_title].index[0]
    content_idx = content_index[movie_title]

    collab_scores = list(enumerate(collab_sim[collab_idx]))
    content_scores = list(enumerate(content_sim[content_idx]))

    hybrid_scores = [(i, (cs[1] + collab_scores[i][1]) / 2) for i, cs in enumerate(content_scores)]
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    seen = set()
    for idx, score in hybrid_scores:
        title = movies.iloc[idx]['title']
        if title != movie_title and title not in seen:
            seen.add(title)
            recommendations.append((title, movies.iloc[idx]['genres'], score))
        if len(recommendations) >= top_n:
            break

    return pd.DataFrame(recommendations, columns=["title", "genres", "score"])

def fetch_tmdb_details(title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {'api_key': TMDB_API_KEY, 'query': title}
    resp = requests.get(url, params=params).json()
    if resp['results']:
        movie = resp['results'][0]
        movie_id = movie['id']
        poster = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None
        overview = movie.get('overview')
        rating = movie.get('vote_average')
        tmdb_link = f"https://www.themoviedb.org/movie/{movie_id}"

        trailer_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
        trailer_resp = requests.get(trailer_url, params={'api_key': TMDB_API_KEY}).json()
        trailer_link = None
        for v in trailer_resp.get('results', []):
            if v['type'] == 'Trailer' and v['site'] == 'YouTube':
                trailer_link = f"https://www.youtube.com/watch?v={v['key']}"
                break

        return {
            'title': movie['title'],
            'poster': poster,
            'overview': overview,
            'rating': rating,
            'tmdb_link': tmdb_link,
            'trailer': trailer_link
        }
    return None

def get_streaming_availability(title, country="AU"):
    try:
        jw = JustWatch(country=country)
        result = jw.search_for_item(query=title)
        offers = result['items'][0].get('offers', [])
        services = list(set(offer['provider_id'] for offer in offers))
        return services
    except Exception:
        return []

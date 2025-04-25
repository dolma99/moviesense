
import streamlit as st
import pandas as pd
from recommender import (
    load_data, build_similarity_matrices,
    hybrid_recommend, fetch_tmdb_details,
    get_streaming_availability
)

st.set_page_config(page_title="🎬 Smart Movie Recommender", layout="wide")
st.title("🎬 Smart Movie Recommender System")

# Load data
movies, ratings = load_data()
collab_sim, content_sim, movie_index, content_index = build_similarity_matrices(movies, ratings)

# Sidebar user preference tracking
st.sidebar.header("🎯 Your Profile")
if "liked" not in st.session_state:
    st.session_state.liked = []

liked_movie = st.sidebar.selectbox("Like a movie?", movies['title'].unique())
if st.sidebar.button("👍 Add to Likes"):
    st.session_state.liked.append(liked_movie)
    st.sidebar.success(f"Added {liked_movie}")

st.sidebar.markdown("### ❤️ Liked Movies")
st.sidebar.write(st.session_state.liked)

# Main recommendation engine
st.subheader("💡 Get Movie Recommendations")
selected_movie = st.selectbox("Select a movie you liked", sorted(movies['title'].unique()))
top_n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("🔍 Recommend"):
    recs = hybrid_recommend(selected_movie, movies, collab_sim, content_sim, movie_index, content_index, top_n)
    for _, row in recs.iterrows():
        with st.container():
            st.markdown(f"## 🎞️ {row['title']} ({row['genres']})")
            details = fetch_tmdb_details(row['title'])
            if details:
                cols = st.columns([1, 3])
                if details['poster']:
                    cols[0].image(details['poster'], width=150)
                cols[1].markdown(f"**TMDb Rating:** ⭐ {details['rating']} / 10")
                cols[1].markdown(f"**Overview:** {details['overview']}")
                if details['trailer']:
                    cols[1].markdown(f"[▶ Watch Trailer]({details['trailer']})")
                cols[1].markdown(f"[🌐 TMDb Link]({details['tmdb_link']})")

            services = get_streaming_availability(row['title'])
            if services:
                st.info(f"🟢 Available on: {', '.join(services)}")
            else:
                st.warning("⚠️ Streaming availability not found.")

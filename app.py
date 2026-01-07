# app.py (FIXED VERSION)
# =============================================================================
# Streamlit App: Hybrid Movie Recommendation System (MovieLens 25M)
# Uses pre-trained models stored in ./models
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Movie Recommender • Hybrid (Content + CF)",
    page_icon="🎬",
    layout="wide",
)

ARTIFACT_DIR = "./models"

# -----------------------------------------------------------------------------
# LOAD ARTIFACTS
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_artifacts():
    files = os.listdir(ARTIFACT_DIR)

    tfidf_file = sorted([f for f in files if "tfidf_vectorizer" in f])[0]
    user_factors_file = sorted([f for f in files if "user_factors" in f])[0]
    movie_factors_file = sorted([f for f in files if "movie_factors" in f])[0]
    mappings_file = sorted([f for f in files if "index_mappings" in f])[0]
    movies_meta_file = sorted([f for f in files if "movies_metadata" in f])[0]

    stats_files = sorted([f for f in files if "model_stats" in f and f.endswith(".json")])
    has_stats = len(stats_files) > 0

    tfidf_vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, tfidf_file))
    user_factors = joblib.load(os.path.join(ARTIFACT_DIR, user_factors_file))
    movie_factors = joblib.load(os.path.join(ARTIFACT_DIR, movie_factors_file))

    with open(os.path.join(ARTIFACT_DIR, mappings_file), "r") as f:
        mappings = json.load(f)
    movies_df = pd.read_csv(os.path.join(ARTIFACT_DIR, movies_meta_file))

    if has_stats:
        with open(os.path.join(ARTIFACT_DIR, stats_files[0]), "r") as f:
            stats = json.load(f)
    else:
        stats = {
            "n_users_total": len(mappings["user_id_map_sampled"]),
            "n_movies_total": len(mappings["movie_idx_to_id_sampled"]),
            "n_ratings_total": 0,
            "cf_mae_sample": 0.0,
            "cf_rmse_sample": 0.0,
            "precision_at_k": None,
            "recall_at_k": None,
            "k_eval": 10,
        }

    user_id_map_sampled = {int(k): int(v) for k, v in mappings["user_id_map_sampled"].items()}
    movie_idx_to_id_sampled = {int(k): int(v) for k, v in mappings["movie_idx_to_id_sampled"].items()}
    movie_id_to_idx_sampled = {v: k for k, v in movie_idx_to_id_sampled.items()}

    return (
        tfidf_vectorizer,
        user_factors,
        movie_factors,
        user_id_map_sampled,
        movie_idx_to_id_sampled,
        movie_id_to_idx_sampled,
        movies_df,
        stats,
    )

(
    tfidf_vectorizer,
    user_factors,
    movie_factors,
    user_id_map_sampled,
    movie_idx_to_id_sampled,
    movie_id_to_idx_sampled,
    movies_df,
    stats,
) = load_artifacts()

# Build TF-IDF matrix (NO caching of unhashable objects)
combined_text = (
    movies_df["title"].fillna("") + " " +
    movies_df["genres"].fillna("").str.replace("|", " ")
)
tfidf_matrix = tfidf_vectorizer.transform(combined_text)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_content_recommendations(movie_id: int, top_n: int = 10) -> pd.DataFrame:
    """Movie-to-movie recommendations using TF-IDF cosine similarity."""
    if movie_id not in movies_df["movieId"].values:
        return pd.DataFrame()

    idx = movies_df[movies_df["movieId"] == movie_id].index[0]
    query_vec = tfidf_matrix[idx]

    sim_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    similar_indices = np.argsort(sim_scores)[::-1][1:top_n+1]
    scores = sim_scores[similar_indices]

    recs = movies_df.iloc[similar_indices].copy()
    recs["content_score"] = scores
    return recs[["movieId", "title", "genres", "content_score"]]


def get_cf_recommendations(user_id: int, top_n: int = 10) -> pd.DataFrame:
    """User-to-movie recommendations using SVD latent factors."""
    if user_id not in user_id_map_sampled:
        return pd.DataFrame()

    user_idx = user_id_map_sampled[user_id]
    user_pred = user_factors[user_idx].dot(movie_factors.T)

    sorted_idx = np.argsort(user_pred)[::-1][:top_n]
    movie_ids = [movie_idx_to_id_sampled[i] for i in sorted_idx]
    scores = [float(np.clip(user_pred[i], 0.5, 5.0)) for i in sorted_idx]

    recs = movies_df[movies_df["movieId"].isin(movie_ids)].copy()
    recs = recs.set_index("movieId").loc[movie_ids].reset_index()
    recs["cf_score"] = scores
    return recs[["movieId", "title", "genres", "cf_score"]]


def get_hybrid_recommendations(user_id: int, movie_id: int, alpha: float = 0.6, top_n: int = 10) -> pd.DataFrame:
    """Hybrid recommendations combining content and CF scores."""
    cb = get_content_recommendations(movie_id, top_n=200)
    cf = get_cf_recommendations(user_id, top_n=500)

    if cb.empty and cf.empty:
        return pd.DataFrame()

    merged = cb.merge(cf[["movieId", "cf_score"]], on="movieId", how="inner")

    if merged.empty:
        merged = cb.copy()
        merged["cf_score"] = np.nan

    def norm(s):
        if s.isna().all():
            return s.fillna(0.0)
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(1.0, index=s.index)
        return (s - mn) / (mx - mn)

    merged["content_norm"] = norm(merged["content_score"])
    merged["cf_norm"] = norm(merged["cf_score"])
    merged["hybrid_score"] = alpha * merged["content_norm"] + (1 - alpha) * merged["cf_norm"]
    merged = merged[merged["movieId"] != movie_id]
    merged = merged.sort_values("hybrid_score", ascending=False).head(top_n)

    return merged[["movieId", "title", "genres", "content_score", "cf_score", "hybrid_score"]]


def render_movie_cards(df: pd.DataFrame, score_col: str | None = None):
    """Render recommendations as responsive cards."""
    if df is None or df.empty:
        st.info("No recommendations to display.")
        return

    for _, row in df.iterrows():
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### 🎬 {row['title']}")
                st.markdown(f"**Genres:** {row['genres'] if pd.notna(row['genres']) else 'N/A'}")
                if score_col and score_col in row and not pd.isna(row[score_col]):
                    st.markdown(f"**Score:** {row[score_col]:.3f}")
            with col2:
                st.image(
                    "https://via.placeholder.com/150x220.png?text=Movie",
                    use_container_width=True,
                )
        st.divider()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------

st.sidebar.title("⚙️ Controls")

mode = st.sidebar.radio(
    "Recommendation Mode",
    ["Content-based", "Collaborative Filtering", "Hybrid"],
)

st.sidebar.markdown("---")

st.sidebar.subheader("Global Model Stats")
st.sidebar.markdown(f"- Users (total): **{stats['n_users_total']}**")
st.sidebar.markdown(f"- Movies (total): **{stats['n_movies_total']}**")
st.sidebar.markdown(f"- Ratings (total): **{stats['n_ratings_total']}**")
st.sidebar.markdown(f"- CF MAE: **{stats['cf_mae_sample']:.3f}**")
st.sidebar.markdown(f"- CF RMSE: **{stats['cf_rmse_sample']:.3f}**")

if stats.get("precision_at_k") is not None:
    st.sidebar.markdown(f"- Precision@{stats['k_eval']}: **{stats['precision_at_k']:.3f}**")
    st.sidebar.markdown(f"- Recall@{stats['k_eval']}: **{stats['recall_at_k']:.3f}**")

st.sidebar.markdown("---")
st.sidebar.caption("Built with MovieLens 25M • Hybrid (TF‑IDF + SVD)")

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------

st.title("🎥 Hybrid Movie Recommendation System")
st.markdown(
    "Delivering personalized movie suggestions using a hybrid of **content-based** "
    "and **collaborative filtering** techniques on the MovieLens 25M dataset."
)

tab1, tab2 = st.tabs(["🔍 Get Recommendations", "📊 Dataset & Model Overview"])

# ------------------ TAB 1: RECOMMENDER UI -----------------------------------

with tab1:
    st.subheader("Get Movie Recommendations")

    movie_titles = movies_df["title"].tolist()
    selected_movie_title = st.selectbox(
        "Select a reference movie (for content-based & hybrid):",
        options=movie_titles,
        index=0,
    )
    selected_movie_id = movies_df.loc[movies_df["title"] == selected_movie_title, "movieId"].iloc[0]

    sample_users = sorted(list(user_id_map_sampled.keys()))[:1000]
    selected_user_id = st.selectbox(
        "Select a user (for CF & hybrid):",
        options=sample_users,
    )

    top_n = st.slider("Number of recommendations", min_value=5, max_value=30, value=10, step=1)

    alpha = st.slider(
        "Content vs CF weight (Hybrid only)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1,
    )

    if st.button("🚀 Recommend"):
        if mode == "Content-based":
            st.markdown("#### Content-based recommendations (similar movies):")
            recs = get_content_recommendations(selected_movie_id, top_n=top_n)
            render_movie_cards(recs, score_col="content_score")

        elif mode == "Collaborative Filtering":
            st.markdown("#### Collaborative filtering recommendations (for this user):")
            recs = get_cf_recommendations(selected_user_id, top_n=top_n)
            render_movie_cards(recs, score_col="cf_score")

        else:  # Hybrid
            st.markdown("#### Hybrid recommendations (content + CF):")
            recs = get_hybrid_recommendations(
                user_id=selected_user_id,
                movie_id=selected_movie_id,
                alpha=alpha,
                top_n=top_n,
            )
            render_movie_cards(recs, score_col="hybrid_score")

# ------------------ TAB 2: OVERVIEW -----------------------------------------

with tab2:
    st.subheader("Dataset & Model Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Dataset Summary")
        st.metric("Total Users", f"{stats['n_users_total']:,}")
        st.metric("Total Movies", f"{stats['n_movies_total']:,}")
        st.metric("Total Ratings", f"{stats['n_ratings_total']:,}")

    with col2:
        st.markdown("##### CF Model Performance")
        st.metric("MAE (sample)", f"{stats['cf_mae_sample']:.3f}")
        st.metric("RMSE (sample)", f"{stats['cf_rmse_sample']:.3f}")
        if stats.get("precision_at_k") is not None:
            st.metric(f"Precision@{stats['k_eval']}", f"{stats['precision_at_k']:.3f}")
            st.metric(f"Recall@{stats['k_eval']}", f"{stats['recall_at_k']:.3f}")

    st.markdown("---")
    st.markdown("##### Random Sample of Movies")
    st.dataframe(movies_df.sample(20).reset_index(drop=True))


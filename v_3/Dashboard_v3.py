import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

from recommender_logic_v3 import (
    get_similar_products_cosine_similarity, 
    recommend_user_via_basket_items_prod_name, 
    recommend_user_via_users_prod_name, 
    get_similar_products_knn_full, 
    recommend_grocery_meta
)

st.set_page_config(layout="wide")
st.title("🛒 Multi-Engine E-Commerce Recommendation Dashboard")


@st.cache_resource
def load_compressed_assets():
    # Dynamically locate the folder where Dashboard_v3.py sits
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(current_dir, 'dashboard_assets.pkl.gz')
    
    # joblib handles .gz decompression natively
    return joblib.load(pkl_path)

# Execute the cache function
assets = load_compressed_assets()

# Unpack Global Metadata
id_to_name = assets['id_to_name']
user_item_matrix = assets['user_item_matrix']

# Unpack Dense Cosine Assets
item_sim_df = assets['item_sim_df']
user_sim_df = assets['user_sim_df']

# Unpack KNN Assets
#model_knn_sample = assets['model_knn_sample']
#item_item_sparse_sample = assets['item_item_sparse_sample']
#matrix_product_ids_sample = assets['matrix_product_ids_sample']

model_knn_full = assets['model_knn_full']
item_item_sparse_full = assets['item_item_sparse_full']
matrix_product_ids_full = assets['matrix_product_ids_full']

# Unpack Amazon Content-Based Assets
df_amazon = assets['df_amazon']
amazon_indices = assets['amazon_indices']
amazon_match_indices = assets['amazon_match_indices']
amazon_match_scores = assets['amazon_match_scores']

# Sidebar Navigation
navigation = st.sidebar.radio("Navigate Engines", [
    "Instacart: Find Similar Products (KNN Full)",
    "Instacart: Find Similar Products (Cosine)",
    "Instacart: Personalized User Recommendations",
    "Amazon: Content-Based Meta Engine"
])

if navigation == "Instacart: Find Similar Products (KNN Full)":
    st.subheader("🚀 High-Performance Item-Item Engine (KNN Matrix)")
    valid_names = [id_to_name[pid] for pid in matrix_product_ids_full if pid in id_to_name]
    chosen_name = st.selectbox("Search Product Catalog:", sorted(valid_names))
    chosen_id = [k for k, v in id_to_name.items() if v == chosen_name][0]
    
    if st.button("Calculate Nearest Neighbors"):
        # Correctly pass arguments into the fixed recommender_logic module
        recs = get_similar_products_knn_full(
            chosen_id, 
            matrix_product_ids_full, 
            model_knn_full, 
            item_item_sparse_full, 
            id_to_name
        )
        st.dataframe(recs, use_container_width=True, column_config={
    "similarity_score": st.column_config.NumberColumn(format="%.4f")})

elif navigation == "Instacart: Find Similar Products (Cosine)":
    st.subheader("📊 Standard Item-Item Engine (Dense Cosine DataFrame)")
    valid_names = [id_to_name[pid] for pid in item_sim_df.keys() if pid in id_to_name]
    chosen_name = st.selectbox("Search Sample Catalog:", sorted(valid_names))
    chosen_id = [k for k, v in id_to_name.items() if v == chosen_name][0]
    
    if st.button("Calculate Similarities"):
        recs = get_similar_products_cosine_similarity(chosen_id, item_sim_df, id_to_name)
        st.dataframe(recs, use_container_width=True)

elif navigation == "Instacart: Personalized User Recommendations":
    st.subheader("👤 User-Centric Recommendation Hub")
    valid_user_ids = user_item_matrix.index.tolist()
    chosen_user = st.selectbox("Select a User ID to inspect:", valid_user_ids)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Option A: Based on their overall purchase history basket similarity**")
        if st.button("Run Basket Engine"):
            st.dataframe(recommend_user_via_basket_items_prod_name(chosen_user, user_item_matrix, item_sim_df, id_to_name), use_container_width=True)
    with col2:
        st.markdown("**Option B: Collaborative Filtering (What similar customers bought)**")
        if st.button("Run User-User Engine"):
            st.dataframe(recommend_user_via_users_prod_name(chosen_user, user_sim_df, user_item_matrix, id_to_name), use_container_width=True)

elif navigation == "Amazon: Content-Based Meta Engine":
    st.subheader("📖 Content text-matching engine (TF-IDF Title Matching)")
    amazon_choices = df_amazon.set_index('parent_asin')['title'].to_dict()
    chosen_title = st.selectbox("Select an Amazon Item:", list(amazon_choices.values()))
    chosen_asin = [k for k, v in amazon_choices.items() if v == chosen_title][0]
    
    if st.button("Find Text Match Recommendations"):
        # Pass the separated indices and score arrays explicitly
        recs = recommend_grocery_meta(chosen_asin, amazon_indices, amazon_match_indices, amazon_match_scores, df_amazon)
        
        if recs.empty:
            st.warning("⚠️ Parent ASIN not found in this dataset slice.")
        else:
            st.dataframe(recs, use_container_width=True)
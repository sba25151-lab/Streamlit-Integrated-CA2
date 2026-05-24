import streamlit as st
import plotly.express as px
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
st.title("Multi-Engine E-Commerce Recommendation Dashboard")


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
    st.subheader("High-Performance Item-Item Engine (KNN Matrix)")
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
            id_to_name)

        st.dataframe(recs, use_container_width=True, column_config={"similarity_score": st.column_config.NumberColumn(format="%.4f")})
       
        st.markdown("### 📊 Recommendation Confidence")
        # Determine the correct name column whether on Amazon or Instacart
        name_col = 'title' if 'title' in recs else 'product_name'
    
        # Create a clean horizontal bar chart using Plotly Express
        fig = px.bar(
        recs, 
        x='similarity_score', 
        y=name_col, 
        orientation='h',
        color='similarity_score',
        color_continuous_scale='Blues')
    
        # Reverse the Y-axis so the highest score is at the top of the chart
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)



elif navigation == "Instacart: Find Similar Products (Cosine)":
    st.subheader("Standard Item-Item Engine (Dense Cosine DataFrame)")
    valid_names = [id_to_name[pid] for pid in item_sim_df.keys() if pid in id_to_name]
    chosen_name = st.selectbox("Search Sample Catalog:", sorted(valid_names))
    chosen_id = [k for k, v in id_to_name.items() if v == chosen_name][0]
    
    if st.button("Calculate Similarities"):
        recs = get_similar_products_cosine_similarity(chosen_id, item_sim_df, id_to_name)
        st.dataframe(recs, use_container_width=True)

        st.markdown("### 📊 Recommendation Confidence")
        # Determine the correct name column whether on Amazon or Instacart
        name_col = 'title' if 'title' in recs else 'product_name'
    
        # Create a clean horizontal bar chart using Plotly Express
        fig = px.bar(
        recs, 
        x='similarity_score', 
        y=name_col, 
        orientation='h',
        color='similarity_score',
        color_continuous_scale='Blues')
    
        # Reverse the Y-axis so the highest score is at the top of the chart
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


elif navigation == "Instacart: Personalized User Recommendations":
    st.subheader("👥 User-Based Personalization Engines")
    valid_user_ids = sorted(list(user_item_matrix.index))
    chosen_user = st.selectbox("Select a User ID to inspect:", valid_user_ids)
    
    st.markdown("### 📈 Customer Purchase Profile")
    
    # Calculate basket size for all users instantly from the user_item_matrix
    all_basket_sizes = (user_item_matrix > 0).sum(axis=1).reset_index()
    all_basket_sizes.columns = ['user_id', 'basket_size']
    
    # Extract metrics for the selected user vs the dataset group average
    chosen_user_size = int(all_basket_sizes[all_basket_sizes['user_id'] == chosen_user]['basket_size'].iloc[0])
    avg_basket_size = float(all_basket_sizes['basket_size'].mean())
    
    # Render side-by-side metric comparison cards
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(
            label=f"Unique Items Bought by User {chosen_user}", 
            value=chosen_user_size,
            delta=f"{chosen_user_size - round(avg_basket_size, 1)} vs Dataset Avg"
        )
    with metric_col2:
        st.metric(
            label="Average Basket Size (All Users)", 
            value=f"{avg_basket_size:.1f} items"
        )
        
    # Generate the distribution histogram using Plotly
    fig_hist = px.histogram(
        all_basket_sizes,
        x='basket_size',
        title="Where Does This User Fall in the Purchase Distribution?",
        labels={'basket_size': 'Number of Unique Products Purchased'},
        color_discrete_sequence=['#2E86C1'],
        nbins=20
    )
    
    # Draw a vertical line showing exactly where the selected user stands
    fig_hist.add_vline(
        x=chosen_user_size, 
        line_dash="dash", 
        line_color="#E67E22", 
        line_width=3,
        annotation_text=f" User {chosen_user} Location", 
        annotation_position="top right"
    )
    
    fig_hist.update_layout(
        xaxis_title="Unique Items Purchased",
        yaxis_title="Number of Customers",
        bargap=0.05
    )
    
    # Display the interactive plot on your dashboard
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Generate Personal Recommendations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Option A: Based on their overall purchase history basket similarity**")
        if st.button("Run Basket Engine"):
            
            recs = recommend_user_via_basket_items_prod_name(chosen_user, user_item_matrix, item_sim_df, id_to_name).reset_index()
            
            if recs.empty:
                st.warning("No recommendations found for this user's profile.")
            else:
                st.dataframe(recs, use_container_width=True)

                st.markdown("### 📊 Recommendation Confidence")
                name_col = 'title' if 'title' in recs else 'product_name'
        
                # Create a clean horizontal bar chart using Plotly Express
                fig = px.bar(
                    recs, 
                    x=name_col, 
                    y='recommendation_score', 
                    orientation='v',
                    color='recommendation_score',
                    color_continuous_scale='Blues'
                )
        
                # Reverse the Y-axis so the highest score is at the top of the chart
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Option B: Collaborative Filtering (What similar customers bought)**")
        if st.button("Run User-User Engine"):
            
            recs = recommend_user_via_users_prod_name(chosen_user, user_sim_df, user_item_matrix, id_to_name).reset_index()
            
            if recs.empty:
                st.warning("No recommendations found for this user's profile.")
            else:
                st.dataframe(recs, use_container_width=True)
                
                st.markdown("### 📊 Recommendation Confidence")
                name_col = 'title' if 'title' in recs else 'product_name'
                
                fig = px.bar(
                    recs, 
                    x=name_col, 
                    y='similar_people_bought', 
                    orientation='v',
                    color='similar_people_bought',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)


elif navigation == "Amazon: Content-Based Meta Engine":
    st.subheader("🛒 Dataset Overview: Top 10 Amazon Stores")

    # Count the items per store, grab the top 10
    top_stores = df_amazon['store'].value_counts().head(10).reset_index()
    top_stores.columns = ['store', 'item_count']

    
    fig_stores = px.bar(
        top_stores, 
        x='store', 
        y='item_count', 
        title="Items per Brand in Database",
        color='item_count',
        color_continuous_scale='Teal')
    st.plotly_chart(fig_stores, use_container_width=True)

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
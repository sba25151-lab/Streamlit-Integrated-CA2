import pandas as pd

# =========================================================================
#   THE MACHINE LEARNING MODEL FUNCTIONS
# =========================================================================

def get_similar_products_cosine_similarity(product_id, item_sim_df, id_to_name, top_n=10):
    if product_id not in item_sim_df.columns:
        return pd.DataFrame(columns=['product_name', 'similarity_score'])
    similarity_scores = item_sim_df[product_id].sort_values(ascending=False).drop(product_id, errors='ignore')
    top_products = similarity_scores.head(top_n)
    return pd.DataFrame({
        'product_name': top_products.index.map(id_to_name),
        'similarity_score': top_products.values
    })

# --- ENGINE 1B: USER RECS VIA PAST BASKET (Instacart Sample) ---
def recommend_user_via_basket_items_prod_name(user_id, user_item_matrix, item_sim_df, id_to_name, top_n=5):
    user_history = user_item_matrix.loc[user_id]
    bought_product_ids = user_history[user_history > 0].index
    similar_scores = item_sim_df[bought_product_ids].sum(axis=1).drop(bought_product_ids, errors='ignore')
    top_recs = similar_scores.sort_values(ascending=False).head(top_n)
    return pd.DataFrame({'product_name': top_recs.index.map(id_to_name), 'recommendation_score': top_recs.values})

# --- ENGINE 2: USER-USER RECS VIA SIMILAR CUSTOMERS (Instacart Sample) ---
def recommend_user_via_users_prod_name(user_id, user_sim_df, user_item_matrix, id_to_name, top_n=5, n_similar_users=10):
    similar_users = user_sim_df[user_id].sort_values(ascending=False).drop(user_id, errors='ignore')
    top_similar_users = similar_users.head(n_similar_users).index
    similar_users_purchases = user_item_matrix.loc[top_similar_users].sum(axis=0)
    already_bought = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = similar_users_purchases.drop(already_bought, errors='ignore')
    top_recs = recommendations.sort_values(ascending=False).head(top_n)
    return pd.DataFrame({'product_name': top_recs.index.map(id_to_name), 'similar_people_bought': top_recs.values})

# --- ENGINE 3: NEAREST NEIGHBORS (Instacart Full Dataset) ---
def get_similar_products_knn_full(product_id, matrix_product_ids_full, model_knn_full, item_item_sparse_full, id_to_name, top_n=10):
    matrix_row_idx = matrix_product_ids_full.get_loc(product_id)
    distances, indices = model_knn_full.kneighbors(item_item_sparse_full[matrix_row_idx], n_neighbors=top_n + 1)
    flat_indices = indices.flatten()[1:]
    flat_distances = distances.flatten()[1:]
    rec_names_df = [id_to_name.get(matrix_product_ids_full[idx], f"Unknown ({matrix_product_ids_full[idx]})") for idx in flat_indices]
    return pd.DataFrame({'product_name': rec_names_df, 'similarity_score': [1 - d for d in flat_distances]})

# --- ENGINE 4: CONTENT-BASED FILTERING (Amazon Grocery Metadata) ---
def recommend_grocery_meta(parent_asin, amazon_indices, amazon_cosine_sim, df_amazon):
    if parent_asin not in amazon_indices:
        return pd.DataFrame(columns=['parent_asin', 'title', 'store'])
    idx = amazon_indices[parent_asin]
    sim_scores = sorted(list(enumerate(amazon_cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    item_indices = [i[0] for i in sim_scores[1:6]]
    return df_amazon.iloc[item_indices]
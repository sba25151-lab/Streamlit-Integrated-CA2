import pandas as pd

# =========================================================================
#   THE MACHINE LEARNING MODEL FUNCTIONS
# =========================================================================

def get_similar_products_cosine_similarity(product_id, item_sim_df, id_to_name, top_n=10):
    if product_id not in item_sim_df.keys():
        return pd.DataFrame(columns=['product_name', 'similarity_score'])
    similarity_scores = item_sim_df[product_id].sort_values(ascending=False).drop(product_id, errors='ignore')
    top_products = similarity_scores.head(top_n)
    return pd.DataFrame({
        'product_name': top_products.index.map(id_to_name),
        'similarity_score': top_products.values
    })

# --- ENGINE 1B: USER RECS VIA PAST BASKET (Instacart Sample) --- (Dictionary-Safe Version) ---
def recommend_user_via_basket_items_prod_name(user_id, user_item_matrix, item_sim_df, id_to_name, top_n=5):
    # Get the items this user has already bought
    user_history = user_item_matrix.loc[user_id]
    bought_product_ids = user_history[user_history > 0].index
    
    # Accumulate similarity scores from the dictionary
    scores = {}
    for pid in bought_product_ids:
        # Check if the product exists in our top-50 similarity dictionary
        if pid in item_sim_df:
            # item_sim_df[pid] is a Pandas Series of its top 50 similar items
            for sim_pid, score in item_sim_df[pid].items():
                scores[sim_pid] = scores.get(sim_pid, 0) + score
                
    # Convert results to a Series, drop items they already bought, and sort
    if not scores:
        return pd.DataFrame(columns=['product_name', 'recommendation_score'])
        
    similar_scores = pd.Series(scores).drop(bought_product_ids, errors='ignore')
    top_recs = similar_scores.sort_values(ascending=False).head(top_n)
    
    return pd.DataFrame({
        'product_name': top_recs.index.map(id_to_name), 
        'recommendation_score': top_recs.values
    })


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
    # 1. Graceful fallback if item doesn't exist
    if parent_asin not in amazon_indices:
        return pd.DataFrame(columns=['parent_asin', 'title', 'store', 'similarity_score'])
        
    # 2. Extract index safely
    idx = amazon_indices[parent_asin]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
        
    # 3. Calculate scores identically to your Jupyter notebook
    try:
        sim_scores = list(enumerate(amazon_cosine_sim[idx]))
    except Exception:
        # Fallback if amazon_cosine_sim has been modified into a dictionary format
        if isinstance(amazon_cosine_sim, dict) and parent_asin in amazon_cosine_sim:
            top_matches_dict = amazon_cosine_sim[parent_asin]
            top_asin_keys = [asin for asin in top_matches_dict.keys() if asin != parent_asin][:5]
            results_df = df_amazon[df_amazon['parent_asin'].isin(top_asin_keys)].copy()
            results_df['similarity_score'] = results_df['parent_asin'].map(top_matches_dict)
            results_df['similarity_score'] = results_df['similarity_score'].astype(float)
            return results_df[['parent_asin', 'title', 'store', 'similarity_score']].reset_index(drop=True)
        return pd.DataFrame(columns=['parent_asin', 'title', 'store', 'similarity_score'])

    # Sort descending 
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 5 matches (skipping index 0)
    top_matches = sim_scores[1:6]
    item_indices = [int(i[0]) for i in top_matches]
    scores = [float(i[1]) for i in top_matches]
    
    # 4. Strictly slice, copy, and align columns to prevent metadata leaks
    results_df = df_amazon[['parent_asin', 'title', 'store']].iloc[item_indices].copy()
    results_df['similarity_score'] = scores
    
    # 5. Reset index completely to clear raw pandas internal row-numbers 
    return results_df.reset_index(drop=True)
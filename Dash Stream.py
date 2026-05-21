import streamlit as st
import pasndas as pd
import numpy as np

st.title("Instacart Product Recommendation Dashboard")

# User Input via Dropdown
product_list = list(id_to_name.values())
chosen_name = st.selectbox("Select a Product:", product_list)

# Find the ID from the name
chosen_id = [k for k, v in id_to_name.items() if v == chosen_name][0]

# Trigger function with a button click
if st.button("Generate Similar Products"):
    st.write(f"Calculating recommendations for ID: {chosen_id}...")
    
    # Run function
    results_df = get_similar_products_knn_full(chosen_id, top_n=10)
    
    # Display the interactive dataframe natively
    st.subheader("Top Recommendations")
    st.dataframe(results_df)
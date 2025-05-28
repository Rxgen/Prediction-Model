import streamlit as st
import pandas as pd
import requests

# Load the cleaned dataset
df = pd.read_csv("ml_ready.csv")

# Required columns
required = ['product_title', 'bsr_movement', 'price_change', 'review_growth',
            'rating_change', 'listing_age_days', 'new_product_flag']

# Validate columns
missing_cols = [col for col in required if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns in CSV: {missing_cols}")
    st.stop()

# App UI setup
st.set_page_config(page_title="Trend Prediction", layout="centered")
st.title("🛍️ Product Trend Prediction Dashboard")

# Dropdown to select a product
selected_title = st.selectbox("Choose a product:", df['product_title'])

# Autofill feature values from selected product
product = df[df['product_title'] == selected_title].iloc[0]

with st.form("predict_form"):
    bsr_movement = st.number_input("BSR Movement", value=float(product['bsr_movement']))
    price_change = st.number_input("Price Change", value=float(product['price_change']))
    review_growth = st.number_input("Review Growth", value=float(product['review_growth']))
    rating_change = st.number_input("Rating Change", value=float(product['rating_change']))
    listing_age_days = st.number_input("Listing Age (days)", value=int(product['listing_age_days']))
    new_product_flag = st.selectbox("Is it a new product?", [0, 1], index=int(product['new_product_flag']))

    submitted = st.form_submit_button("Predict Trend")

if submitted:
    payload = {
        "bsr_movement": bsr_movement,
        "price_change": price_change,
        "review_growth": review_growth,
        "rating_change": rating_change,
        "listing_age_days": listing_age_days,
        "new_product_flag": new_product_flag
    }

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=payload)
        result = response.json()

        if "trend_probability" in result:
            st.success(f"📈 Trend Probability: {result['trend_probability']}%")
        else:
            st.error("❌ Prediction failed. Please check the API or input values.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

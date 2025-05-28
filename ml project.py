import pandas as pd

# Load your CSV
df = pd.read_csv(r'C:\Users\ankur\OneDrive\Desktop\ml\Combine data.csv')

# Step 2: Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Step 3: Keep only the required columns
df = df[['title', 'brand', 'categories', 'final_price', 'bs_rank', 'rating', 'reviews_count', 'date_first_available']]
df = df.rename(columns={
    'title': 'product_title',
    'final_price': 'price',
    'bs_rank': 'bsr',
    'reviews_count': 'reviews',
    'date_first_available': 'listing_date'
})

# Step 4: Drop rows with missing values in key columns
df = df.dropna(subset=['price', 'bsr', 'listing_date'])

# Step 5: Convert listing_date to datetime
df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
df = df.dropna(subset=['listing_date'])

# Optional: Preview the cleaned data
print(df.head())

#Feature Engineering Code
import numpy as np
from datetime import datetime

# Step 1: Calculate listing age in days
df['listing_age_days'] = (pd.to_datetime('today') - df['listing_date']).dt.days

# Step 2: Create a new product flag (1 if listed in last 30 days)
df['new_product_flag'] = np.where(df['listing_age_days'] <= 30, 1, 0)

# Step 3: Simulate placeholder values (replace with real historical logic when available)
np.random.seed(42)  # For consistent results

df['bsr_movement'] = np.random.randint(-1000, 1000, size=df.shape[0])          # Simulated change in BSR
df['price_change'] = np.random.uniform(-100, 100, size=df.shape[0])           # Simulated price difference
df['review_growth'] = np.random.uniform(-10, 50, size=df.shape[0])            # Simulated review growth rate
df['rating_change'] = np.random.uniform(-1, 1, size=df.shape[0])              # Simulated rating change

# Optional: Preview new features
print(df[['product_title', 'listing_age_days', 'new_product_flag', 'bsr_movement', 'price_change', 'review_growth', 'rating_change']].head())

# Define label based on simulated logic
# A product is marked as trending if:
# - BSR improved significantly (bsr_movement < -200)
# - Review count has grown notably (review_growth > 5)

df['trend_label'] = np.where((df['bsr_movement'] < -200) & (df['review_growth'] > 5), 1, 0)

# Optional: Check label distribution
print(df['trend_label'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Step 1: Define feature columns and label
features = ['bsr_movement', 'price_change', 'review_growth', 'rating_change', 'listing_age_days', 'new_product_flag']
X = df[features]
y = df['trend_label']

# Step 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Initialize models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42)
}

# Step 4: Train models
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} model trained.")

from sklearn.metrics import classification_report, roc_auc_score

# Step: Evaluate all models
for name, model in models.items():
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]  # For AUC
    
    print(f"\n🔍 Model: {name}")
    print(classification_report(y_test, preds))
    print(f"ROC AUC Score: {roc_auc_score(y_test, proba):.4f}")

import joblib

# Step 1: Save the best model (choose based on evaluation results)
best_model = models['RandomForest']  # Change to 'XGBoost' or 'LightGBM' if better
joblib.dump(best_model, 'trend_prediction_model.pkl')
print("✅ Model saved as 'trend_prediction_model.pkl'")

# Step 2: Predict trend probabilities for all products
df['trend_probability'] = best_model.predict_proba(X)[:, 1]

# Step 3: Save top 100 trending products to CSV
df[['product_title', 'trend_probability']].sort_values(by='trend_probability', ascending=False).head(100).to_csv('top_trending_products.csv', index=False)
print("📁 Top 100 trending products saved to 'top_trending_products.csv'")

import pandas as pd
import joblib

# Load your data and model
df = pd.read_csv("ml.csv")
model = joblib.load("trend_prediction_model.pkl")

# Prepare features (ensure columns match what the model expects)
features = ['bsr_movement', 'price_change', 'review_growth', 'rating_change', 'listing_age_days', 'new_product_flag']
X = df[features]

# Predict probabilities
df['trend_probability'] = model.predict_proba(X)[:, 1]

# Save top trending products
df[['product_title', 'trend_probability']].sort_values(by='trend_probability', ascending=False).head(100).to_csv("top_trending_products.csv", index=False)

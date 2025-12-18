import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. Load Data & Merge ---
print("1. Loading and merging data...")
try:
    # Files are read from the 'data/' subfolder
    train_df = pd.read_csv('data/train.csv')
    meal_df = pd.read_csv('data/meal_info.csv')
    center_df = pd.read_csv('data/fulfilment_center_info.csv')
except FileNotFoundError:
    print("Error: Ensure your data files are in the 'data/' folder and named correctly.")
    exit()

df = train_df.merge(meal_df, on='meal_id', how='left')
df = df.merge(center_df, on='center_id', how='left')
print(f"Initial shape of combined data: {df.shape}")

# --- 2. Feature Engineering (Integrating Contextual Factors) ---
print("2. Creating features (Time-Series and Categorical)...")

# A. Time-based Features (to capture seasonality/academic calendar effects)
df['quarter'] = (df['week'] // 13) + 1
df['semester'] = (df['week'] // 26) + 1

# B. Lag Features (Recent demand is the best predictor of future demand)
df['lag_1_num_orders'] = df.groupby(['center_id', 'meal_id'])['num_orders'].shift(1)
df.dropna(subset=['lag_1_num_orders'], inplace=True)

# C. Categorical Feature Encoding (for model use)
categorical_cols = ['category', 'cuisine', 'center_type']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# D. Drop Unnecessary Columns
cols_to_drop = ['id', 'week', 'center_id', 'meal_id', 'base_price']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Separate features (X) and target (y)
X = df.drop('num_orders', axis=1)
y = df['num_orders']
y_log = np.log1p(y) # Log-transform target for better model performance

print(f"Shape after feature engineering: {X.shape}")

# --- 3. Split Data (Time-Series Split) ---
# Use the last 20% of data for the test set (ensures latest weeks are for validation)
split_point = int(len(X) * 0.8)
X_train = X.iloc[:split_point]
y_train = y_log.iloc[:split_point]
X_test = X.iloc[split_point:]
y_test = y.iloc[split_point:]       # Original target used for final RMSE calculation

print(f"Train data size: {X_train.shape[0]}, Test data size: {X_test.shape[0]}")

# --- 4. Train XGBoost Regressor ---
print("4. Training XGBoost Model (This may take a minute)...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    tree_method='hist'
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, np.log1p(y_test))],
    verbose=False
)

print("XGBoost Model Training Complete.")

# --- 5. Evaluate and Save Model ---
y_pred_log = xgb_model.predict(X_test)
y_pred = np.expm1(y_pred_log) # Inverse log transform
y_pred[y_pred < 0] = 0
y_pred = np.round(y_pred) # Orders must be integers

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Final Model RMSE (Root Mean Squared Error): {rmse:.2f}")

# Save the trained model to the root folder
# The Streamlit app will load this file.
xgb_model.save_model('trained_model.json')
print("Model saved successfully as 'trained_model.json' in the root directory.")
import os
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define the directory path where your files are stored
directory_path = r'C:\Users\chris.mooney\PycharmProjects\M&V\AccountsWithData'

# Get all files that start with 'Variables' in the specified directory
file_list = [f for f in os.listdir(directory_path)]

# Dropdown to select the file
selected_file = st.selectbox("Select a Customer", file_list)

# Load the selected data file
file_path = os.path.join(directory_path, selected_file)
df = pd.read_csv(file_path)

# Preprocessing the data
df = df.dropna(subset=['usage'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.weekday + 1
df['weekend'] = df['day_of_week'].isin([6, 7]).astype(int)

# Feature groups
weather_features = ['temperature', 'cloud_cover', 'humidity', 'wind_speed',
                    'wind_direction', 'dew_point', 'precipitation',
                    'solar_radiation', 'pressure']
time_features = ['month', 'day_of_week', 'weekend', 'hour', 'holiday']

# --- UI Layout ---
st.title("XGBoost Prediction vs. Actual")
st.markdown("## Select Variables:")
use_all_weather = st.checkbox("Select All Weather Variables", value=True)
use_all_time = st.checkbox("Select All Time Variables", value=True)

# --- Weather Group ---
selected_weather = []
st.markdown("### Weather Variables")
weather_cols = st.columns(3)
for i, feature in enumerate(weather_features):
    default = use_all_weather
    if not use_all_weather:
        default = False
    if weather_cols[i % 3].checkbox(feature, value=default, key=f"weather_{feature}"):
        selected_weather.append(feature)

# --- Time Group ---
selected_time = []
st.markdown("### Time Variables")
time_cols = st.columns(3)
for i, feature in enumerate(time_features):
    default = use_all_time
    if not use_all_time:
        default = False
    if time_cols[i % 3].checkbox(feature, value=default, key=f"time_{feature}"):
        selected_time.append(feature)

# --- Combine Selected Features ---
selected_features = selected_weather + selected_time

# --- Year-Based Train/Test Split ---
st.markdown("## Select Training and Prediction Periods")

# Get available years from the data
available_years = sorted(df['timestamp'].dt.year.unique())

# Multiselect for training years
train_years = st.multiselect("Select Training Year(s)", available_years, default=[available_years[0]])

# Date input for prediction start date
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()
prediction_start = st.date_input("Select Prediction Start Date", min_value=min_date, max_value=max_date, value=min_date)

# Split data
df_train = df[df['timestamp'].dt.year.isin(train_years)]
df_predict = df[df['timestamp'].dt.date > prediction_start]


# --- Model Training ---
if selected_features:
    # Define features and target
    X_train = df_train[selected_features]
    y_train = df_train['usage']

    X_pred = df_predict[selected_features]
    y_pred_actual = df_predict['usage']

    # Validation split
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Model training
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(X_train_split, y_train_split)

    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    st.subheader(f"Validation RMSE: {rmse:.2f} kWh")

    # --- Feature Importance as Percentage (Gain) ---
    importance_dict = model.get_booster().get_score(importance_type='gain')
    # Convert to DataFrame
    importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['gain'])
    importance_df['percentage'] = 100 * importance_df['gain'] / importance_df['gain'].sum()
    importance_df = importance_df.sort_values('percentage', ascending=True)
    # Plot
    fig1, ax1 = plt.subplots(figsize=(8, max(4, 0.3 * len(importance_df))))
    bars = ax1.barh(importance_df.index, importance_df['percentage'])
    ax1.set_xlabel('Importance (%)')
    ax1.set_title('Feature Importance')
    # Add percentage labels to end of bars
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{width:.1f}%', va='center', fontsize=9)
    st.pyplot(fig1)

    # Actual vs Predicted (on prediction set)
    y_pred = model.predict(X_pred)
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_pred_actual, y_pred, alpha=0.5, edgecolors='k')
    ax2.plot([y_pred_actual.min(), y_pred_actual.max()],
             [y_pred_actual.min(), y_pred_actual.max()], 'r--')
    ax2.set_xlabel('Actual Usage (kWh)')
    ax2.set_ylabel('Predicted Usage (kWh)')
    ax2.set_title(f'Prediction Set: Actual vs. Predicted Usage (After {prediction_start})')
    st.pyplot(fig2)
else:
    st.warning("Select at least one variable.")


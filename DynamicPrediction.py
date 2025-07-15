import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Set wide layout
st.set_page_config(page_title="Electricity Usage Prediction", layout="wide")

# Define the directory path where your files are stored
directory_path = r'C:\Users\chris.mooney\PycharmProjects\M&V\AccountsWithData'

# Get all files that start with 'Variables' in the specified directory
file_list = [f for f in os.listdir(directory_path)]

# Dropdown to select the file
selected_file = st.selectbox("Select a Customer", file_list)

# Load the selected data file
file_path = os.path.join(directory_path, selected_file)
df = pd.read_csv(file_path)

# Preprocessing
df = df.dropna(subset=['usage'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.weekday + 1
df['weekend'] = df['day_of_week'].isin([6, 7]).astype(int)

# Features and target
weather_features = ['temperature', 'cloud_cover', 'humidity', 'wind_speed',
                    'wind_direction', 'dew_point', 'precipitation',
                    'solar_radiation', 'pressure']
time_features = ['month', 'day_of_week', 'weekend', 'hour', 'holiday']

# Train/predict split
df_train = df[df['timestamp'].dt.year == 2024]
df_predict = df[df['timestamp'].dt.year > 2022]

features = weather_features + time_features
X_train = df_train[features]
y_train = df_train['usage']

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# Predict
X_predict = df_predict[features]
df_predict = df_predict.copy()
df_predict['predicted_usage'] = model.predict(X_predict)

# Combine
df_combined = pd.concat([
    df_train[['timestamp', 'usage', 'temperature']],
    df_predict[['timestamp', 'usage', 'predicted_usage', 'temperature']]
])

# --- User Options ---
min_date = df_combined['timestamp'].min().date()
max_date = df_combined['timestamp'].max().date()

st.markdown("## Select Date Range")
start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

if start_date > end_date:
    st.error("End date must be after start date.")
    st.stop()

# Toggle options
st.markdown("### Data to Show")
show_actual = st.checkbox("Show Actual Usage", value=True)
show_predicted = st.checkbox("Show Predicted Usage", value=True)
show_temperature = st.checkbox("Show Temperature (°F)", value=True)

# Filter
mask = (df_combined['timestamp'].dt.date >= start_date) & (df_combined['timestamp'].dt.date <= end_date)
df_filtered = df_combined[mask]

# --- Plotting ---
fig, ax1 = plt.subplots(figsize=(16, 8), dpi=150)

# Primary axis
if show_actual:
    ax1.plot(df_filtered['timestamp'], df_filtered['usage'], label="Actual Usage", color='blue', alpha=0.7)
if show_predicted and 'predicted_usage' in df_filtered:
    ax1.plot(df_filtered['timestamp'], df_filtered['predicted_usage'], label="Predicted Usage", color='orange', linestyle='--', alpha=0.7)

ax1.set_xlabel("Timestamp")
ax1.set_ylabel("Electric Usage", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Temperature on secondary axis
if show_temperature:
    ax2 = ax1.twinx()
    ax2.plot(df_filtered['timestamp'], df_filtered['temperature'], label="Temperature", color='red', alpha=0.5)
    ax2.set_ylabel("Temperature (°F)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
else:
    ax2 = None

# Legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title("Actual vs. Predicted Usage")
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
st.pyplot(fig, use_container_width=True)

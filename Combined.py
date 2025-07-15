import os
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- File Selection ---
directory_path = r'C:\Users\chris.mooney\PycharmProjects\M&V\AccountsWithData'
file_list = [f for f in os.listdir(directory_path)]
selected_file = st.selectbox("Select a Customer", file_list)
file_path = os.path.join(directory_path, selected_file)
df = pd.read_csv(file_path)

# --- Preprocessing ---
df = df.dropna(subset=['usage'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.weekday + 1
df['weekend'] = df['day_of_week'].isin([6, 7]).astype(int)

# --- Feature Groups ---
weather_features = ['temperature', 'cloud_cover', 'humidity', 'wind_speed',
                    'wind_direction', 'dew_point', 'precipitation',
                    'solar_radiation', 'pressure']
time_features = ['month', 'day_of_week', 'weekend', 'hour', 'holiday']

# --- UI: Feature Selection ---
st.title("M&V Prediction vs. Actual")
st.markdown("### Select Variables:")

selected_weather = []
st.markdown("##### Weather Variables")
use_all_weather = st.checkbox("Select All Weather Variables", value=True)
weather_cols = st.columns(5)
for i, feature in enumerate(weather_features):
    default = use_all_weather
    if weather_cols[i % 5].checkbox(feature, value=default, key=f"weather_{feature}"):
        selected_weather.append(feature)

selected_time = []
st.markdown("##### Time Variables")
use_all_time = st.checkbox("Select All Time Variables", value=True)
time_cols = st.columns(5)
for i, feature in enumerate(time_features):
    default = use_all_time
    if time_cols[i % 5].checkbox(feature, value=default, key=f"time_{feature}"):
        selected_time.append(feature)

selected_features = selected_weather + selected_time

# --- Train/Test Period Selection ---
st.markdown("### Select Training and Prediction Periods")
available_years = sorted(df['timestamp'].dt.year.unique())
train_years = st.multiselect("Select Training Year(s)", available_years, default=[available_years[0]])
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()
prediction_start = st.date_input("Select Prediction Start Date", min_value=min_date, max_value=max_date, value=min_date)

df_train = df[df['timestamp'].dt.year.isin(train_years)]
df_predict = df[df['timestamp'].dt.date > prediction_start]

# --- Model Training & Evaluation ---
if selected_features:
    X_train = df_train[selected_features]
    y_train = df_train['usage']
    X_pred = df_predict[selected_features]
    y_pred_actual = df_predict['usage']

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(X_train_split, y_train_split)

    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    st.subheader(f"Validation RMSE: {rmse:.2f} kWh")

    # --- Feature Importance ---
    importance_dict = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['gain'])
    importance_df['percentage'] = 100 * importance_df['gain'] / importance_df['gain'].sum()
    importance_df = importance_df.sort_values('percentage', ascending=True)

    fig1, ax1 = plt.subplots(figsize=(8, max(4, 0.3 * len(importance_df))))
    bars = ax1.barh(importance_df.index, importance_df['percentage'])
    ax1.set_xlabel('Importance (%)')
    ax1.set_title('Feature Importance')
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', va='center', fontsize=9)
    st.pyplot(fig1)

    # --- Predict on Prediction Set ---
    y_pred = model.predict(X_pred)
    df_predict = df_predict.copy()
    df_predict['predicted_usage'] = y_pred

    # --- Scatter Plot ---
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_pred_actual, y_pred, alpha=0.5, edgecolors='k')
    ax2.plot([y_pred_actual.min(), y_pred_actual.max()],
             [y_pred_actual.min(), y_pred_actual.max()], 'r--')
    ax2.set_xlabel('Actual Usage (kWh)')
    ax2.set_ylabel('Predicted Usage (kWh)')
    ax2.set_title(f'Prediction Set: Actual vs. Predicted Usage (After {prediction_start})')
    st.pyplot(fig2)

    # --- Date Range for Time-Series ---
    st.markdown("## Select Date Range for Time-Series Plot")
    ts_min = df_predict['timestamp'].min().date()
    ts_max = df_predict['timestamp'].max().date()
    start_date = st.date_input("Start date", min_value=ts_min, max_value=ts_max, value=ts_min, key='ts_start')
    end_date = st.date_input("End date", min_value=ts_min, max_value=ts_max, value=ts_max, key='ts_end')

    if start_date > end_date:
        st.error("End date must be after start date.")
        st.stop()

    mask = (df_predict['timestamp'].dt.date >= start_date) & (df_predict['timestamp'].dt.date <= end_date)
    df_filtered = df_predict[mask]

    # --- Time-Series Toggle UI ---
    st.markdown("## Time-Series Plot Options")
    show_actual = st.checkbox("Show Actual Usage", value=True)
    show_predicted = st.checkbox("Show Predicted Usage", value=True)
    show_temperature = st.checkbox("Show Temperature", value=True)

    # --- Time-Series Plot ---
    fig3, ax1 = plt.subplots(figsize=(16, 8), dpi=150)

    if show_actual and 'usage' in df_filtered:
        ax1.plot(df_filtered['timestamp'], df_filtered['usage'], label="Actual Usage", color='blue', alpha=0.7)
    if show_predicted and 'predicted_usage' in df_filtered:
        ax1.plot(df_filtered['timestamp'], df_filtered['predicted_usage'], label="Predicted Usage", color='orange', linestyle='--', alpha=0.7)

    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Electric Usage (kWh)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    if show_temperature and 'temperature' in df_filtered.columns:
        ax2 = ax1.twinx()
        ax2.plot(df_filtered['timestamp'], df_filtered['temperature'], label="Temperature", color='red', alpha=0.5)
        ax2.set_ylabel("Temperature (°F)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    else:
        ax2 = None

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title("Actual vs. Predicted Usage with Temperature Overlay")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)

else:
    st.warning("Select at least one variable.")

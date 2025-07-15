import os
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime

# File Selection
directory_path = r'C:\Users\chris.mooney\PycharmProjects\M&V\AccountsWithData'
file_list = [f for f in os.listdir(directory_path)]
selected_file = st.selectbox("Select a Customer", file_list)
file_path = os.path.join(directory_path, selected_file)
df = pd.read_csv(file_path)

# Preprocessing
df = df.dropna(subset=['usage'])
df['usage'] = df['usage'] * 4  # <-- Multiply by 4 to change from kWh to kW
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.weekday + 1
df['weekend'] = df['day_of_week'].isin([6, 7]).astype(int)

# Variable Selections
weather_features = ['temperature', 'cloud_cover', 'humidity', 'wind_speed', 'wind_direction', 'dew_point', 'precipitation', 'solar_radiation', 'pressure']
time_features = ['month', 'day_of_week', 'weekend', 'hour', 'holiday']

# UI: Title/Header
st.title("M&V Prediction vs. Actual")
st.markdown("### Select Variables:")

# UI: Weather Variable Checkboxes
selected_weather = []
st.markdown("##### Weather Variables")
use_all_weather = st.checkbox("Select All Weather Variables", value=True)
weather_cols = st.columns(5)
for i, feature in enumerate(weather_features):
    default = use_all_weather
    if weather_cols[i % 5].checkbox(feature, value=default, key=f"weather_{feature}"):
        selected_weather.append(feature)

# UI: Time Variable Checkboxes
selected_time = []
st.markdown("##### Time Variables")
use_all_time = st.checkbox("Select All Time Variables", value=True)
time_cols = st.columns(5)
for i, feature in enumerate(time_features):
    default = use_all_time
    if time_cols[i % 5].checkbox(feature, value=default, key=f"time_{feature}"):
        selected_time.append(feature)

# Combine All Selected Variables
selected_features = selected_weather + selected_time

# Train/Test Period Selection
st.markdown("### Select Training and Prediction Periods")

# Set default date range
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()

# Selection box for date range option
date_range_option = st.selectbox(
    "Choose Training Period Option",
    options=["Custom Range", "Prior 12 Months", "Prior 24 Months"]
)

# Determine start and end dates based on selection
prediction_buffer_days = 7  # Leave a gap for prediction data

if date_range_option == "Prior 12 Months":
    adjusted_max_date = max_date - datetime.timedelta(days=prediction_buffer_days)
    start_date = adjusted_max_date - datetime.timedelta(days=365)
    start_date = max(start_date, min_date)
    end_date = adjusted_max_date
    st.info(f"Using Prior 12 Months: {start_date.strftime('%m-%d-%Y')} to {end_date.strftime('%m-%d-%Y')} (prediction starts after {end_date.strftime('%m-%d-%Y')})")



elif date_range_option == "Prior 24 Months":
    prediction_buffer_days = 7  # Leave a gap for prediction data
    adjusted_max_date = max_date - datetime.timedelta(days=prediction_buffer_days)
    start_date = adjusted_max_date - datetime.timedelta(days=730)
    start_date = max(start_date, min_date)
    end_date = adjusted_max_date
    st.info(f"Using Prior 24 Months: {start_date.strftime('%m-%d-%Y')} to {end_date.strftime('%m-%d-%Y')} (prediction starts after {end_date.strftime('%m-%d-%Y')})")



else:  # Custom Range
    start_date, end_date = st.slider(
        "Select ML Training Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="MM-DD-YYYY"
    )

# Filter data within selected date range
df_train = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
df_predict = df[df['timestamp'].dt.date > end_date]

# Model Training & Evaluation
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
    st.subheader(f"Total RMSE: {rmse:.2f} kW")

    # Add predictions and actuals to a DataFrame with day info
    val_results = X_val.copy()
    val_results['actual'] = y_val
    val_results['predicted'] = y_val_pred
    val_results['day_of_week'] = df_train.loc[X_val.index, 'day_of_week']

    # Map day numbers to names
    day_map = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    val_results['day_name'] = val_results['day_of_week'].map(day_map)

    # Compute RMSE per day
    rmse_by_day = (
        val_results.groupby('day_name', group_keys=False)
        .apply(lambda x: np.sqrt(mean_squared_error(x['actual'], x['predicted'])))
        .round(2)
    )

    # Sort days correctly
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    rmse_by_day = rmse_by_day.reindex(ordered_days)

    # Display in markdown format
    st.markdown("#### RMSE by Day")
    for day, rmse_val in rmse_by_day.items():
        st.markdown(f"**{day}**: {rmse_val} kW")

    # --- Feature Importance ---
    importance_dict = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['gain'])
    importance_df['percentage'] = 100 * importance_df['gain'] / importance_df['gain'].sum()
    importance_df = importance_df.sort_values('percentage', ascending=True)
    fig1, ax1 = plt.subplots(figsize=(6, max(3, 0.3 * len(importance_df))))
    bars = ax1.barh(importance_df.index, importance_df['percentage'])
    ax1.set_xlabel('Importance (%)')
    ax1.set_title('Variable Importance')
    ax1.set_xlim(right=importance_df['percentage'].max() + 8)  # Add padding to the right
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', va='center', fontsize=9)
    st.pyplot(fig1)

    # --- Predict on Prediction Set ---
    y_pred = model.predict(X_pred)
    df_predict = df_predict.copy()
    df_predict['predicted_usage'] = y_pred

    # --- Scatter Plot ---
    # Filter df_predict to only include timestamps after end_date
    scatter_df = df_predict[df_predict['timestamp'].dt.date > end_date].copy()

    if scatter_df.empty:
        st.warning("No prediction data available after training period for scatter plot.")
    else:
        y_pred_actual_filtered = scatter_df['usage']
        y_pred_filtered = scatter_df['predicted_usage']

        fig2, ax2 = plt.subplots()
        ax2.scatter(y_pred_actual_filtered, y_pred_filtered, alpha=0.5, edgecolors='k')
        ax2.plot([y_pred_actual_filtered.min(), y_pred_actual_filtered.max()],
                 [y_pred_actual_filtered.min(), y_pred_actual_filtered.max()], 'r--')
        ax2.set_xlabel('Actual Usage (kWh)')
        ax2.set_ylabel('Predicted Usage (kWh)')
        ax2.set_title(f"Actual vs. Predicted Usage (After {end_date.strftime('%m-%d-%Y')})")
        st.pyplot(fig2)

    # Date Range Slider for Time-Series
    st.markdown("## Select Date Range for Time-Series Plot")
    ts_min = (df_predict[df_predict['timestamp'].dt.date > end_date]['timestamp'].min()).date()
    ts_max = df_predict['timestamp'].max().date()
    start_date, end_date = st.slider(
        "Select Time-Series Date Range",
        min_value=ts_min,
        max_value=ts_max,
        value=(ts_min, ts_max),
        format="MM-DD-YYYY"
    )

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
    fig3, ax1 = plt.subplots(figsize=(16, 8), dpi=300)

    if show_actual and 'usage' in df_filtered:
        ax1.plot(df_filtered['timestamp'], df_filtered['usage'], label="Actual Usage", color='blue', alpha=0.7)
    if show_predicted and 'predicted_usage' in df_filtered:
        ax1.plot(df_filtered['timestamp'], df_filtered['predicted_usage'], label="Predicted Usage", color='orange', linestyle='--', alpha=0.7)

    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Electric Usage (kW)", color='blue')
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

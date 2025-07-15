import os
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime

# --- File Selection ---
directory_path = r'C:\Users\chris.mooney\PycharmProjects\M&V\AccountsWithData'
file_list = [f for f in os.listdir(directory_path)]
selected_file = st.selectbox("Select a Customer", file_list)
file_path = os.path.join(directory_path, selected_file)
df = pd.read_csv(file_path)

# --- Preprocessing ---
df = df.dropna(subset=['usage'])
df['usage'] *= 4  # Scale usage early
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.weekday + 1
df['weekend'] = df['day_of_week'].isin([6, 7]).astype(int)

# --- Variable Selections ---
weather_features = ['temperature', 'cloud_cover', 'humidity', 'wind_speed', 'wind_direction', 'dew_point', 'precipitation', 'solar_radiation', 'pressure']
time_features = ['month', 'day_of_week', 'weekend', 'hour', 'holiday']

st.title("M&V Prediction vs. Actual")
st.markdown("### Select Variables:")

# --- UI: Weather Variables ---
selected_weather = []
st.markdown("##### Weather Variables")
use_all_weather = st.checkbox("Select All Weather Variables", value=True)
weather_cols = st.columns(5)
for i, feature in enumerate(weather_features):
    default = use_all_weather
    if weather_cols[i % 5].checkbox(feature, value=default, key=f"weather_{feature}"):
        selected_weather.append(feature)

# --- UI: Time Variables ---
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

min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()
prediction_buffer_days = 7
adjusted_max_date = max_date - datetime.timedelta(days=prediction_buffer_days)

date_range_option = st.selectbox(
    "Choose Training Period Option",
    options=["Custom Range", "Prior 12 Months", "Prior 24 Months"]
)

if date_range_option == "Prior 12 Months":
    start_date = max(min_date, adjusted_max_date - datetime.timedelta(days=365))
    end_date = adjusted_max_date
    st.info(f"Using Prior 12 Months: {start_date.strftime('%m-%d-%Y')} to {end_date.strftime('%m-%d-%Y')} (prediction starts after {end_date.strftime('%m-%d-%Y')})")

elif date_range_option == "Prior 24 Months":
    start_date = max(min_date, adjusted_max_date - datetime.timedelta(days=730))
    end_date = adjusted_max_date
    st.info(f"Using Prior 24 Months: {start_date.strftime('%m-%d-%Y')} to {end_date.strftime('%m-%d-%Y')} (prediction starts after {end_date.strftime('%m-%d-%Y')})")

else:
    start_date, end_date = st.slider(
        "Select Training Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, adjusted_max_date),
        format="MM-DD-YYYY"
    )

# --- Split data ---
df_train = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
df_predict = df[df['timestamp'].dt.date > end_date]

# --- Model Training ---
if selected_features and not df_train.empty and not df_predict.empty:
    X_train = df_train[selected_features]
    y_train = df_train['usage']
    X_pred = df_predict[selected_features]
    y_pred_actual = df_predict['usage']

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(X_train_split, y_train_split)

    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    st.subheader(f"Total RMSE: {rmse:.2f} kWh")

    # RMSE by Day
    val_results = X_val.copy()
    val_results['actual'] = y_val
    val_results['predicted'] = y_val_pred
    val_results['day_of_week'] = df_train.loc[X_val.index, 'day_of_week']

    day_map = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    val_results['day_name'] = val_results['day_of_week'].map(day_map)

    rmse_by_day = (
        val_results.groupby('day_name', group_keys=False)
        .apply(lambda x: np.sqrt(mean_squared_error(x['actual'], x['predicted'])))
        .round(2)
    )
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    rmse_by_day = rmse_by_day.reindex(ordered_days)

    st.markdown("#### RMSE by Day")
    for day, rmse_val in rmse_by_day.items():
        st.markdown(f"**{day}**: {rmse_val} kWh")

    # --- Feature Importance ---
    importance_dict = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['gain'])
    importance_df['percentage'] = 100 * importance_df['gain'] / importance_df['gain'].sum()
    importance_df = importance_df.sort_values('percentage', ascending=True)

    fig1, ax1 = plt.subplots(figsize=(6, max(3, 0.3 * len(importance_df))))
    bars = ax1.barh(importance_df.index, importance_df['percentage'])
    ax1.set_xlabel('Importance (%)')
    ax1.set_title('Variable Importance')
    ax1.set_xlim(right=importance_df['percentage'].max() + 8)
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', va='center', fontsize=9)
    st.pyplot(fig1)

    # --- Predict ---
    y_pred = model.predict(X_pred)
    df_predict = df_predict.copy()
    df_predict['predicted_usage'] = y_pred

    # --- Shared Slider for Time-Series and Scatter ---
    st.markdown("## Select Date Range for Evaluation Plots")

    ts_min = df_predict[df_predict['timestamp'].dt.date > end_date]['timestamp'].min()
    ts_max = df_predict['timestamp'].max()

    if pd.isna(ts_min) or pd.isna(ts_max):
        st.warning("No prediction data available after the training period.")
        st.stop()

    ts_min = ts_min.date()
    ts_max = ts_max.date()

    plot_start_date, plot_end_date = st.slider(
        "Select Prediction Date Range",
        min_value=ts_min,
        max_value=ts_max,
        value=(ts_min, ts_max),
        format="MM-DD-YYYY"
    )

    mask = (df_predict['timestamp'].dt.date >= plot_start_date) & (df_predict['timestamp'].dt.date <= plot_end_date)
    df_filtered = df_predict[mask]

    # --- Scatter Plot ---
    if df_filtered.empty:
        st.warning("No prediction data in selected range for scatter plot.")
    else:
        fig2, ax2 = plt.subplots()
        ax2.scatter(df_filtered['usage'], df_filtered['predicted_usage'], alpha=0.5, edgecolors='k')
        ax2.plot([df_filtered['usage'].min(), df_filtered['usage'].max()],
                 [df_filtered['usage'].min(), df_filtered['usage'].max()], 'r--')
        ax2.set_xlabel('Actual Usage (kWh)')
        ax2.set_ylabel('Predicted Usage (kWh)')
        ax2.set_title(f"Actual vs. Predicted Usage ({plot_start_date.strftime('%m-%d-%Y')} to {plot_end_date.strftime('%m-%d-%Y')})")
        st.pyplot(fig2)

    # --- Time-Series Plot Toggle ---
    st.markdown("## Time-Series Plot Options")
    show_actual = st.checkbox("Show Actual Usage", value=True)
    show_predicted = st.checkbox("Show Predicted Usage", value=True)
    show_temperature = st.checkbox("Show Temperature", value=True)

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
    st.warning("Select at least one variable and ensure data exists for both training and prediction.")

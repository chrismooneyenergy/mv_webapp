import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load CSV
file_path = r"C:\Users\chris.mooney\PycharmProjects\M&V\AccountsWithData\Reed & Graham Inc.csv"
df = pd.read_csv(file_path, parse_dates=['timestamp'])

# ---- Feature Engineering ----

# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Cooling and Heating Degree Hours
df['CDH'] = df['temperature'].apply(lambda x: max(0, x - 65))
df['HDH'] = df['temperature'].apply(lambda x: max(0, 65 - x))

# ---- Train-Test Split ----
# (We'll just train on the full set for normalization, no future prediction)
features = [
    'temperature', 'humidity', 'cloud_cover', 'dew_point',
    'wind_speed', 'solar_radiation', 'precipitation', 'pressure',
    'CDH', 'HDH', 'hour', 'day_of_week', 'weekend', 'holiday', 'month'
]

df = df.dropna(subset=features + ['usage'])  # Drop rows with missing inputs

X = df[features]
y = df['usage']

# Train a simple model
print("Start")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
print("End")

# Predict actual usage (optional)
df['predicted_usage'] = model.predict(X)

# ---- Create "Normal Weather" Profile ----
weather_cols = [
    'temperature', 'humidity', 'cloud_cover', 'dew_point',
    'wind_speed', 'solar_radiation', 'precipitation', 'pressure'
]

# Get average weather for each hour/month combo
normal_weather = df.groupby(['hour', 'month'])[weather_cols].mean().reset_index()

# Merge with original data
df = df.merge(normal_weather, on=['hour', 'month'], suffixes=('', '_normal'))

# Recalculate CDH/HDH for normal conditions
df['CDH_normal'] = df['temperature_normal'].apply(lambda x: max(0, x - 65))
df['HDH_normal'] = df['temperature_normal'].apply(lambda x: max(0, 65 - x))

# Temporarily overwrite the actual weather variables with the normal ones for prediction
df_temp = df.copy()

# Replace actual weather with normal weather for prediction
for col in weather_cols:
    df_temp[col] = df_temp[col + '_normal']

# Replace CDH/HDH
df_temp['CDH'] = df_temp['CDH_normal']
df_temp['HDH'] = df_temp['HDH_normal']

# Predict using the original feature names
df['normalized_usage'] = model.predict(df_temp[features])


# ---- Plot Actual vs Normalized Usage ----

# Sort by time for clean plotting
df = df.sort_values(by='timestamp')

plt.figure(figsize=(15, 6))
plt.plot(df['timestamp'], df['usage'], label='Actual Usage', color='blue', linewidth=0.7)
plt.plot(df['timestamp'], df['normalized_usage'], label='Normalized Usage', color='orange', linestyle='--', linewidth=0.7)

plt.title('Actual vs Weather-Normalized Electricity Usage (15-Min Intervals)')
plt.xlabel('Timestamp')
plt.ylabel('kWh')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load the original dataset (same as used for training)
file_path = r'C:\Users\chris.mooney\PycharmProjects\M&V\Variables_550 Battery St_2491340090.csv'
df = pd.read_csv(file_path)

# Drop rows with missing usage values
df = df.dropna(subset=['usage'])

# Step 2: Convert timestamp to datetime and extract features (same preprocessing)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.weekday + 1  # Monday = 1, Sunday = 7
df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [6, 7] else 0)

# Step 3: Add a new column (Modify as needed)
df['new_factor'] = 1  # Example placeholder (modify with actual calculation)

# Step 4: Split the data into training (2022-2023) and prediction (after 2023)
df_train = df[(df['timestamp'].dt.year == 2022)]
df_predict = df[df['timestamp'].dt.year > 2022]

# Step 5: Define the independent variables (including new weather features)
X_train = df_train[['month','day_of_week', 'weekend', 'hour', 'holiday']]

# Define the target variable (usage)
y_train = df_train['usage']

# Step 6: Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Step 7: Train the XGBoost model with updated features
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# Step 8: Prepare the features for prediction (after 2023)
X_predict = df_predict[['month', 'day_of_week', 'weekend', 'hour', 'holiday']]

# Step 9: Make predictions using the trained model
df_predict['predicted_usage'] = model.predict(X_predict)

# Step 10: Save the updated dataset with predictions (optional)
df_predict.to_csv(r'C:\Users\chris.mooney\PycharmProjects\M&V\Predicted_Cranbrook_7137448595.csv', index=False)

# Step 11: Combine the original and predicted data for plotting
df_combined = pd.concat([
    df_train[['timestamp', 'usage', 'temperature']],
    df_predict[['timestamp', 'usage', 'predicted_usage', 'temperature']]
])

# Step 12 (Updated): Plot actual usage vs predicted usage over time, with temperature overlay
plt.figure(figsize=(12, 6))

# Primary y-axis for usage
ax1 = plt.gca()
ax1.plot(df_combined['timestamp'], df_combined['usage'], label="Actual Usage", linestyle='-', alpha=0.7, color='blue')
ax1.plot(df_predict['timestamp'], df_predict['predicted_usage'], label="Predicted Usage", linestyle='--', alpha=0.7, color='orange')
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Electric Usage')
ax1.tick_params(axis='y', labelcolor='blue')

# Secondary y-axis for temperature
ax2 = ax1.twinx()
ax2.plot(df_combined['timestamp'], df_combined['temperature'], label="Temperature", linestyle='-', alpha=0.5, color='red')
ax2.set_ylabel('Temperature (°C)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Title and legend
plt.title('Actual vs Predicted Electric Usage with Temperature Overlay (2022–2025)')

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
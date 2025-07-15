import requests
import pandas as pd
import holidays

# File paths
file_path = r"C:\Users\chris.mooney\PycharmProjects\M&V\Original_Cranbrook_7137448595.csv"
output_path = r"C:\Users\chris.mooney\PycharmProjects\M&V\Variables_Cranbrook_7137448595.csv"


def get_coordinates(zip_code, country_code="US"):
    """Fetch latitude and longitude for a given ZIP code."""
    url = f"http://api.zippopotam.us/{country_code}/{zip_code}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        latitude = float(data["places"][0]["latitude"])
        longitude = float(data["places"][0]["longitude"])
        return latitude, longitude
    else:
        print("Error fetching coordinates. Check ZIP code.")
        return None, None


def get_weather_data(zip_code, start, end):
    """Fetch weather data at hourly intervals and expand to 15-minute segments."""
    latitude, longitude = get_coordinates(zip_code)
    if latitude is None or longitude is None:
        return None

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}"
        f"&start_date={start.date()}&end_date={end.date()}"
        f"&hourly=temperature_2m,cloudcover,relative_humidity_2m,windspeed_10m,winddirection_10m,"
        f"dewpoint_2m,precipitation,shortwave_radiation,surface_pressure"
        f"&temperature_unit=fahrenheit&timezone=auto"
    )

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "hourly" in data:
            timestamps = pd.to_datetime(data["hourly"]["time"])
            df_weather = pd.DataFrame({
                "timestamp": timestamps,
                "temperature": data["hourly"]["temperature_2m"],
                "cloud_cover": data["hourly"]["cloudcover"],
                "humidity": data["hourly"]["relative_humidity_2m"],
                "wind_speed": data["hourly"]["windspeed_10m"],
                "wind_direction": data["hourly"]["winddirection_10m"],
                "dew_point": data["hourly"]["dewpoint_2m"],
                "precipitation": data["hourly"]["precipitation"],
                "solar_radiation": data["hourly"]["shortwave_radiation"],
                "pressure": data["hourly"]["surface_pressure"]
            })
            df_weather.set_index("timestamp", inplace=True)

            # Align to CSV timestamps
            df_weather_15min = df_weather.reindex(df_weather.index.repeat(4))
            df_weather_15min.index = pd.date_range(start=start, end=end, freq='15T')

            df_weather_15min.reset_index(inplace=True)
            df_weather_15min.rename(columns={"index": "timestamp"}, inplace=True)

            return df_weather_15min
        else:
            print("No weather data available for this period.")
            return None
    else:
        print("Error fetching weather data:", response.json())
        return None


def merge_weather_with_data(file_path, zip_code, output_path):
    """Merge weather data with existing CSV and export."""
    df_existing = pd.read_csv(file_path)

    if "timestamp" not in df_existing.columns:
        print("Error: CSV file must contain a 'timestamp' column.")
        return

    df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"])
    start = df_existing["timestamp"].min()
    end = df_existing["timestamp"].max()

    df_weather = get_weather_data(zip_code, start, end)

    if df_weather is not None:
        df_merged = pd.merge(df_existing, df_weather, on="timestamp", how="left")

        df_merged['day_of_week'] = df_merged['timestamp'].dt.weekday + 1
        df_merged['weekend'] = df_merged['day_of_week'].apply(lambda x: 1 if x in [6, 7] else 0)
        df_merged['hour'] = df_merged['timestamp'].dt.hour
        df_merged['month'] = df_merged['timestamp'].dt.month
        us_holidays = holidays.US()
        df_merged['holiday'] = df_merged['timestamp'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)

        df_merged.to_csv(output_path, index=False)
        print(f"Updated file saved to: {output_path}")
    else:
        print("Weather data could not be retrieved. No updates made.")


# Example usage
merge_weather_with_data(file_path, zip_code="94621", output_path=output_path)
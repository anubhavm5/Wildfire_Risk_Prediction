import pandas as pd
import requests
import os

def fetch_weather_data(lat, lon, start_date, end_date):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,relative_humidity_2m_mean,precipitation_sum,windspeed_10m_max"
        f"&timezone=UTC"
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame({
            "date": data["daily"]["time"],
            "temperature": data["daily"]["temperature_2m_max"],
            "humidity": data["daily"]["relative_humidity_2m_mean"],
            "wind_speed": data["daily"]["windspeed_10m_max"],
            "rainfall": data["daily"]["precipitation_sum"]
        })
        df["date"] = pd.to_datetime(df["date"])
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return pd.DataFrame()

def fetch_fire_data_csv(folder_path="C:/Users/anubh/Documents/WildFire_Risk_Prediction/Data_Fire/2024"):
    combined_df = pd.DataFrame()
    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' not found. No fire data loaded.")
        return combined_df

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            df['acq_date'] = pd.to_datetime(df['acq_date'])
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")

    return combined_df

def load_data():
    fire_df = fetch_fire_data_csv()
    if fire_df.empty:
        raise ValueError("No fire data found!")

    fire_df = fire_df.rename(columns={"acq_date": "date"})
    fire_days = fire_df["date"].dt.date.unique()
    start_date, end_date = str(min(fire_days)), str(max(fire_days))

    weather_df = fetch_weather_data(lat=20.5937, lon=78.9629, start_date=start_date, end_date=end_date)
    if weather_df.empty:
        raise ValueError("No weather data found!")

    # Label fire days = 1, others = 0
    weather_df["fire_occurred"] = weather_df["date"].dt.date.isin(fire_days).astype(int)

    return weather_df

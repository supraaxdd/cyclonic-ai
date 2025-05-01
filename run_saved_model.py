from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import keras

SEQ_LEN = 24 # 24 Hours
SAVED_MODEL_PATH = "saved/model.keras"

def read_input_data(input_set: str):
    df = pd.read_json(input_set)
    return pd.json_normalize(df["hourly"])

def get_features(df: pd.DataFrame):
    features = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "temperature_2m": df["temperature_2m"],
        "temperature_80m": df["temperature_80m"],
        "temperature_120m": df["temperature_120m"],
        "temperature_180m": df["temperature_180m"],
        "soil_temperature_0cm": df["soil_temperature_0cm"],
        "soil_temperature_6cm": df["soil_temperature_6cm"],
        "soil_temperature_18cm": df["soil_temperature_18cm"],
        "soil_temperature_54cm": df["soil_temperature_54cm"],
        "pressure": df["surface_pressure"],
    })

    return features

def plot_prediction(prediction):
    plt.figure(figsize=(10, 5))

    plt.plot(prediction, label="Predicted")
    plt.title("Wind Speed Prediction with LSTM")
    
    plt.xlabel("Time Steps")
    plt.ylabel("Scaled Wind Speed")

    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = read_input_data("./input/result-forecast.json")
    features = get_features(df)

    scaler = StandardScaler()

    # Drop non-numeric column
    dates = features["date"]
    scaled = scaler.fit_transform(features.drop(columns=["date"]))

    # Create LSTM-compatible input
    SEQ_LEN = 24
    X = []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
    X = np.array(X)

    # Load and predict
    model = keras.models.load_model(SAVED_MODEL_PATH)
    prediction = model.predict(X).flatten()

    # Optional: trim dates to match prediction shape
    trimmed_dates = dates.iloc[SEQ_LEN:]
    plot_prediction(pd.Series(prediction, index=trimmed_dates))
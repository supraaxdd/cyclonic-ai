import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib

import keras

SEQ_LEN = 24 # 24 Hours
SAVED_MODEL_PATH = "saved/model.keras"
SAVED_X_SCALER = "saved/X_scaler.pkl"
SAVED_Y_SCALER = "saved/y_scaler.pkl"

def read_input_data(input_set: str):
    df = pd.read_json(input_set)
    return pd.json_normalize(df["hourly"])

def get_features(df: pd.DataFrame):
    df["date"] = pd.to_datetime(df["date"])
    df["pressure_delta_3h"] = df["surface_pressure"].diff(periods=3)
    df["temperature_2m_delta_3h"] = df["temperature_2m"].diff(periods=3)

    features = pd.DataFrame({
        "temperature_2m": df["temperature_2m"],
        "temperature_80m": df["temperature_80m"],
        "temperature_120m": df["temperature_120m"],
        "temperature_180m": df["temperature_180m"],
        "soil_temperature_0cm": df["soil_temperature_0cm"],
        "soil_temperature_6cm": df["soil_temperature_6cm"],
        "soil_temperature_18cm": df["soil_temperature_18cm"],
        "soil_temperature_54cm": df["soil_temperature_54cm"],
        "pressure": df["surface_pressure"],
        "pressure_delta_3h": df["pressure_delta_3h"],
        "temperature_2m_delta_3h": df["temperature_2m_delta_3h"]
    })

    return features, df["date"]

def plot_prediction(dates, prediction):
    plt.figure(figsize=(10, 5))

    plt.plot(dates, prediction, label="Predicted Wind Speed")

    plt.title("Predicted Wind Speed")

    plt.xlabel("Date")
    plt.ylabel("Wind Speed (km/h)")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    df = read_input_data("./input/result.json")
    features, dates = get_features(df)

    # Load scalers
    X_scaler = joblib.load(SAVED_X_SCALER)
    y_scaler = joblib.load(SAVED_Y_SCALER)

    # Scale inputs
    scaled = X_scaler.transform(features)

    # Create LSTM-compatible input
    SEQ_LEN = 24
    X = []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
    X = np.array(X)

    trimmed_dates = dates.iloc[SEQ_LEN:]

    # Load and predict
    model = keras.models.load_model(SAVED_MODEL_PATH)
    y_pred_scaled = model.predict(X).flatten()
    y_pred_unscaled = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    plot_prediction(trimmed_dates, y_pred_unscaled)
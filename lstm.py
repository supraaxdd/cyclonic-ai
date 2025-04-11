import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense

SEQUENCE_LENGTH = 24  # 24 hours

def read_input_data(input_set: str):
    df = pd.read_json(input_set)
    return pd.json_normalize(df["hourly"])

def get_vectors(df: pd.DataFrame):
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

    targets = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "wind_speed_10m": df["wind_speed_10m"],
        "wind_speed_80m": df["wind_speed_80m"],
        "wind_speed_120m": df["wind_speed_120m"],
        "wind_speed_180m": df["wind_speed_180m"],
        "wind_direction_10m": df["wind_direction_10m"]
    })

    return features, targets

def plot_lstm_results(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("Wind Speed Prediction with LSTM")
    plt.xlabel("Time Steps")
    plt.ylabel("Scaled Wind Speed")
    plt.legend()
    plt.tight_layout()
    plt.show()

def predict_wind_speed_lstm(features: pd.DataFrame, targets: pd.DataFrame):
    df = features.copy()
    df["wind_speed_10m"] = targets["wind_speed_10m"].values
    df["wind_speed_80m"] = targets["wind_speed_80m"].values
    df["wind_speed_120m"] = targets["wind_speed_120m"].values
    df["wind_speed_180m"] = targets["wind_speed_180m"].values
    df.dropna(inplace=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[["temperature_2m", "temperature_80m", "temperature_120m", "temperature_180m", "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm", "pressure", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m"]])

    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled)):
        X.append(scaled[i - SEQUENCE_LENGTH:i, :-1])
        y.append(scaled[i, -1])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define LSTM model
    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Train
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Predict
    y_pred = model.predict(X_test).flatten()

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nLSTM Model Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    plot_lstm_results(y_test, y_pred)

if __name__ == "__main__":
    df = read_input_data("./input/result.json")
    labels, targets = get_vectors(df)
    predict_wind_speed_lstm(labels, targets)

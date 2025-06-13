import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math

from pathlib import Path
import joblib

from pandas import Series, DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout
from keras.src.optimizers import RMSprop
from keras.src.callbacks import EarlyStopping

SEQUENCE_LENGTH = 24  # 24 hours
SAVED_MODEL_PATH = "saved/model.keras"
SAVED_X_SCALER = "saved/X_scaler.pkl"
SAVED_Y_SCALER = "saved/y_scaler.pkl"

def read_input_data(input_set: str):
    df = pd.read_json(input_set)
    return pd.json_normalize(df["hourly"])

def read_base_info(input_set: str):
    return pd.read_json(input_set)[["lat", "long", "elevation"]].iloc[0]

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
        "wind_speed_10m": df["wind_speed_10m"]
    })

    return features, targets

def calculate_air_density(temp_c: int, pavg: int):
    """
    temp_c = Temperature (°C)
    pavg = The average pressure between the two points (Pa)
    """
    rd = 287.05  # Gas constant for dry air
    kelvin = temp_c + 273.15
    return pavg / (rd * kelvin)

def calculate_distance(x: tuple, y: tuple):
    """
    Returns dy and dx in meters between two (lat, lon) points
    """
    lat_x, lon_x = x
    lat_y, lon_y = y

    deg_to_m_lat = 111000  # meters per degree latitude
    avg_lat_rad = math.radians((lat_x + lat_y) / 2)
    deg_to_m_lon = 111000 * math.cos(avg_lat_rad)

    dx = (lon_y - lon_x) * deg_to_m_lon  # East-West
    dy = (lat_y - lat_x) * deg_to_m_lat  # North-South

    return dy, dx

def calculate_pgf_components(dy: float, dx: float, pd: float, rho: float):
    """
    Calculates the PGF vector components given dy, dx, pressure delta, and air density
    """
    r = math.sqrt(dx**2 + dy**2)
    dP_dx = pd / r * (dx / r)
    dP_dy = pd / r * (dy / r)

    F_x = -1 / rho * dP_dx
    F_y = -1 / rho * dP_dy
    return F_y, F_x  # (dy, dx) for consistency

def calculate_pgf_time_series(info_1, info_2, features_1: pd.DataFrame, features_2: pd.DataFrame) -> pd.DataFrame:
    loc_1 = (info_1["lat"], info_1["long"])
    loc_2 = (info_2["lat"], info_2["long"])
    dy, dx = calculate_distance(loc_2, loc_1)

    results = []

    for i in range(len(features_1)):
        try:
            temp_c = float(features_1["temperature_2m"].iloc[i])
            p1 = float(features_2["pressure"].iloc[i])
            p2 = float(features_1["pressure"].iloc[i])
            pd_val = p2 - p1
            pavg_val = (p1 + p2) / 2
            rho = calculate_air_density(temp_c, pavg_val)
            Fy, Fx = calculate_pgf_components(dy, dx, pd_val, rho)
            magnitude = math.sqrt(Fx**2 + Fy**2)

            results.append({
                "date": features_1["date"].iloc[i],
                "PGF_x": Fx,
                "PGF_y": Fy,
                "PGF_magnitude": magnitude
            })
        except Exception as e:
            print(f"Skipped row {i} due to error: {e}")

    return pd.DataFrame(results)


def plot_lstm_results(y_true, y_pred):
    plt.figure(figsize=(10, 5))

    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")

    plt.title("Wind Speed Prediction with LSTM")

    plt.xlabel("Time Steps")
    plt.ylabel("Wind Speed (km/h)")

    plt.legend()

    plt.tight_layout()

    plt.show()

def build_model(X, y) -> Sequential:
    # Define LSTM model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])

    # Define the optimizer for the LSTM
    optimizer = RMSprop(learning_rate=0.0005)

    model.compile(optimizer=optimizer, loss='mse')
    model.summary()

    return model

def predict_wind_speed_lstm(features: pd.DataFrame, targets: pd.DataFrame):
    df = features.copy()

    df["wind_speed_10m"] = targets["wind_speed_10m"].values

    # Simulating pressure systems moving in and out of the location along with changes in temperature for a given location every 3 hours
    df["pressure_delta_3h"] = df["pressure"].diff(periods=3)
    df["temperature_2m_delta_3h"] = df["temperature_2m"].diff(periods=3)

    df.dropna(inplace=True)

    FEATURE_COLUMNS = [
        "temperature_2m", "temperature_80m", "temperature_120m", "temperature_180m",
        "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm",
        "pressure", "pressure_delta_3h", "temperature_2m_delta_3h",
        "PGF_x", "PGF_y", "PGF_magnitude"
    ]


    # Fit and save X-scaler
    X_scaler = StandardScaler()
    scaled_X = X_scaler.fit_transform(df[FEATURE_COLUMNS])
    joblib.dump(X_scaler, SAVED_X_SCALER)

    # Fit and save y-scaler
    y_scaler = StandardScaler()
    scaled_y = y_scaler.fit_transform(df[["wind_speed_10m"]])
    joblib.dump(y_scaler, SAVED_Y_SCALER)

    # Creating a snapshot of each hour to be predicted based on the previous 24 hours of data
    # for the time-series model
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_X)):
        X.append(scaled_X[i - SEQUENCE_LENGTH:i]) # Snapshot of the input data over the last 24 hours for a given period
        y.append(scaled_y[i][0]) # Compliling what the wind is looking like at this moment (speed and direction)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define a callback to discontinue model training if no further improvements are found
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True
    )

    model = build_model(X, y)

    # Train
    model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[early_stop]
    )

    model.save(Path(SAVED_MODEL_PATH))

    # Predict and unscale
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_unscaled = y_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate
    print(f"\nLSTM Model Evaluation:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_scaled)):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred_scaled):.4f}")

    plot_lstm_results(y_test_unscaled, y_pred)

if __name__ == "__main__":
    df_1 = read_input_data("./input/result.json")
    df_2 = read_input_data("./input/result_2.json")

    info_1 = read_base_info("./input/result.json")
    info_2 = read_base_info("./input/result_2.json")

    features_1, targets_1 = get_vectors(df_1)
    features_2, targets_2 = get_vectors(df_2)

    pgf_df = calculate_pgf_time_series(info_1, info_2, features_1, features_2)
    features_1 = features_1.merge(pgf_df, on="date", how="inner")

    predict_wind_speed_lstm(features_1, targets_1)

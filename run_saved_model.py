import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib
import keras
import json

from lstm import (
    read_input_data,
    read_base_info,
    get_vectors,
    calculate_pgf_time_series
)

SEQ_LEN = 24  # 24 Hours
SAVED_MODEL_PATH = "saved/model.keras"
SAVED_X_SCALER = "saved/X_scaler.pkl"
SAVED_Y_SCALER = "saved/y_scaler.pkl"

def get_features_with_pgf(df_main: pd.DataFrame, df_ref: pd.DataFrame, info_main, info_ref):
    features_main, _ = get_vectors(df_main)
    features_ref, _ = get_vectors(df_ref)

    # Calculate PGF
    pgf_df = calculate_pgf_time_series(info_main, info_ref, features_main, features_ref)
    features = features_main.merge(pgf_df, on="date", how="inner")

    # Deltas
    features["pressure_delta_3h"] = features["pressure"].diff(periods=3)
    features["temperature_2m_delta_3h"] = features["temperature_2m"].diff(periods=3)

    features.dropna(inplace=True)

    return features, features["date"]

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
    # Load inputs for both locations
    df_main = read_input_data("./input/result.json")
    df_ref = read_input_data("./input/result_2.json")

    info_main = read_base_info("./input/result.json")
    info_ref = read_base_info("./input/result_2.json")

    # Get features with PGF
    features, dates = get_features_with_pgf(df_main, df_ref, info_main, info_ref)

    # Feature columns used in training
    FEATURE_COLUMNS = [
        "temperature_2m", "temperature_80m", "temperature_120m", "temperature_180m",
        "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm",
        "pressure", "pressure_delta_3h", "temperature_2m_delta_3h",
        "PGF_x", "PGF_y", "PGF_magnitude"
    ]

    # Load scalers
    X_scaler = joblib.load(SAVED_X_SCALER)
    y_scaler = joblib.load(SAVED_Y_SCALER)

    # Scale input
    scaled = X_scaler.transform(features[FEATURE_COLUMNS])

    # Prepare LSTM sequences
    X = []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
    X = np.array(X)

    trimmed_dates = dates.iloc[SEQ_LEN:]

    # Load model and predict
    model = keras.models.load_model(SAVED_MODEL_PATH)
    y_pred_scaled = model.predict(X).flatten()
    y_pred_unscaled = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    with open("output/prediction.json", "w") as f:
        d = {
            "dates": trimmed_dates.astype(str).to_list(),
            "predictions": y_pred_unscaled.tolist()
        }
        
        f.write(json.dumps(d))

    # Plot
    plot_prediction(trimmed_dates, y_pred_unscaled)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from pandas import DataFrame, Series

plotting = True

SEQUENCE_LENGTH = 24 # 24 Hours (1 Day)

def read_input_data(input_set: str):
    df = pd.read_json(input_set)
    return pd.json_normalize(df["hourly"])

def read_base_info(input_set: str):
    return pd.read_json(input_set)[["lat", "long", "elevation"]]

def get_vectors(df: DataFrame):
    # Compile the labels
    labels = DataFrame({
        "date": pd.to_datetime(df["date"]),
        "temperature_2m": df["temperature_2m"],
        "soil_temperature_0cm": df["soil_temperature_0cm"],
        "pressure": df["surface_pressure"]
    })

    # Compile the features
    features = DataFrame({
        'date': pd.to_datetime(df["date"]),
        'wind_speed_10m': df["wind_speed_10m"],
        'wind_direction_10m': df["wind_direction_10m"]
    }) 

    return labels, features

def plot_features(df: DataFrame):
    fig, axes = plt.subplots(2, figsize=(9, 7))

    axes[0].plot(df["date"], df["temperature_2m"], label="Temperature at 2m")
    axes[0].plot(df["date"], df["soil_temperature_0cm"], label="Soil Temperature at 0cm")
    axes[0].set_title("Temperatures")

    axes[1].plot(df["date"], df["pressure"], label="Atmospheric Pressure (hPa)")
    axes[1].set_title("Atmospheric Pressure")

    # Formatting x-axis
    for ax in axes:
        ax.set_xlim([df["date"].iloc[0], df["date"].iloc[-1]])
        ax.set_xticks([df["date"].iloc[0], df["date"].iloc[-1]])
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d-%H:%M'))
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    axes[0].set_ylabel("Temperature (*C)")
    axes[1].set_ylabel("Atmospheric Pressure (hPa)")

    axes[0].legend()
    axes[1].legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_model_results(results):
    names = [r[0] for r in results]
    y_tests = [r[1] for r in results]
    y_preds = [r[2] for r in results]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for i in range(5):
        axes[i].plot(y_tests[i], label="Actual")
        axes[i].plot(y_preds[i], label="Predicted")
        axes[i].set_title(names[i])
        axes[i].legend(loc="upper right")

    axes[5].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_model_accuracies(results):
    names = [r[0] for r in results]
    rmses = [r[4] for r in results]
    
    plt.figure(figsize=(8, 5))
    plt.barh(names, rmses, color='skyblue')
    plt.xlabel("RMSE (lower is better)")
    plt.title("Model Comparison for Wind Speed Prediction")
    plt.tight_layout()
    plt.show()

def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest Regressor": RandomForestRegressor(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "Linear Regression": LinearRegression(),
        "Support Vector Regressor": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        results.append((name, y_test, y_pred, mse, rmse, mae))
        
    return sorted(results, key=lambda x: x[4]) # Sort by RMSE

def predict_wind_speeds(features: DataFrame, targets: DataFrame):
    # Merge features and targets by date    
    data = features.copy()
    data["wind_speed_10m"] = targets["wind_speed_10m"].values

    data.dropna(inplace=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[["temperature_2m", "soil_temperature_0cm", "pressure", "wind_speed_10m"]])

    X, y = [], []

    for i in range(SEQUENCE_LENGTH, len(scaled)):
        flattened_features = scaled[i - SEQUENCE_LENGTH:i, :-1].flatten()
        X.append(flattened_features)
        y.append(scaled[i, -1])

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    results = evaluate_models(X_train, X_test, y_train, y_test)

    if plotting:
        plot_model_accuracies(results)
        plot_model_results(results)

if __name__ == "__main__":
    df = read_input_data("./input/result.json")
    features, targets = get_vectors(df)

    if plotting:
        plot_features(features)

    predict_wind_speeds(features, targets)
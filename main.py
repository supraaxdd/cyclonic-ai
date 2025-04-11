import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pandas import DataFrame, Series

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
	
if __name__ == "__main__":
    df = read_input_data("./input/result.json")
    features, targets = get_vectors(df)

    plotting = True

    if plotting:
        plot_features(features)
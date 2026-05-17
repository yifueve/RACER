import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")
matplotlib.rcParams.update({"font.size": 16})

SCRIPT_DIR = Path(__file__).resolve().parent

location_labels = {
    "AT": "Austria",
    "AU-NSW": "Sydney/AU",
    "AU-VIC": "Melbourne/AU",
    "BE": "Belgium",
    "BR-SP": "Sao Paulo/BR",
    "CA-ON": "Toronto/CA",
    "CH": "Switzerland",
    "CL-SIC": "Santiago/CL",
    "DE-LU": "Frankfurt/DE",
    "ES": "Spain",
    "FR": "France",
    "IN-WE": "Mumbai/IN",
    "JP-TK": "Tokyo/JP",
    "KR": "Seoul/KR",
    "NL": "Netherlands",
    "PT": "Portugal",
    "SG": "Singapore/SG",
    "US-CAL-CISO": "San Francisco/USA",
    "US-MIDA-PJM": "Philadelphia/USA",
    "US-NY-NYIS": "New York/USA",
    "US-TEX-ERCO": "Dallas/USA",
    "ZA": "Johannesburg/ZA",
}


def load_standardized_prices(data_dir):
    frames = []
    for region_code in sorted(os.listdir(data_dir)):
        region_path = data_dir / region_code
        if not region_path.is_dir():
            continue

        for year_path in sorted(region_path.iterdir()):
            if not year_path.is_dir():
                continue

            file_path = year_path / f"{region_code}_electricity_prices_{year_path.name}.csv"
            if not file_path.is_file():
                continue

            df = pd.read_csv(
                file_path,
                usecols=["Datetime (UTC)", "Price (USD/MWh)"],
            )
            utc = df["Datetime (UTC)"].astype(str)
            df["date"] = utc.str.slice(5, 10)
            # Some standardized regions, such as IN-WE, are stamped at HH:30.
            # Floor to the hour so every region shares the same 24-point axis.
            df["time_of_day"] = utc.str.slice(11, 13) + ":00"
            df["location"] = location_labels.get(region_code, region_code)
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No standardized electricity price CSV files found in {data_dir}")

    return pd.concat(frames, ignore_index=True)


def plot_lines(data, x_col, y_col, xlabel, ylabel, title, output_path, tick_count):
    linestyles = ["-", "--", "-."]
    plt.figure(figsize=(12, 6))

    for i, location in enumerate(data["location"].unique()):
        subset = data[data["location"] == location]
        plt.plot(
            subset[x_col],
            subset[y_col],
            label=location,
            color=f"C{i % 10}",
            linestyle=linestyles[i % len(linestyles)],
            alpha=0.9,
        )

    x_values = data[x_col].unique()
    xticks_indices = np.linspace(0, len(x_values) - 1, num=tick_count, dtype=int)
    plt.xticks(x_values[xticks_indices], rotation=45)
    plt.xlim(0, len(x_values) - 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


prices = load_standardized_prices(SCRIPT_DIR)

mean_daily_prices = (
    prices.groupby(["date", "location"], sort=True)["Price (USD/MWh)"]
    .mean()
    .reset_index()
)
plot_lines(
    mean_daily_prices,
    x_col="date",
    y_col="Price (USD/MWh)",
    xlabel="Day of Year (MM-DD)",
    ylabel="Average Electricity Price (USD/MWh)",
    title="Typical Daily Electricity Price Trends Across Locations",
    output_path=SCRIPT_DIR / "electricity_price_daily_trends.png",
    tick_count=12,
)

mean_time_of_day_prices = (
    prices.groupby(["time_of_day", "location"], sort=True)["Price (USD/MWh)"]
    .mean()
    .reset_index()
)
plot_lines(
    mean_time_of_day_prices,
    x_col="time_of_day",
    y_col="Price (USD/MWh)",
    xlabel="Time of Day (HH:MM) (UTC)",
    ylabel="Average Electricity Price (USD/MWh)",
    title="Typical Daily Electricity Price Variation Across Locations",
    output_path=SCRIPT_DIR / "electricity_price_daily_variation.png",
    tick_count=24,
)

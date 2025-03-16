import pandas as pd
import settings as s
from astral import LocationInfo
from astral.sun import sun
from meteostat import Point, Hourly
import logging

logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """
    Reads the CSV file specified in settings, applies time-based reindexing,
    computes additional features, merges weather data, and returns a prepared DataFrame.

    The following transformations are applied:
        1. Reading & parsing raw data into a DataFrame.
        2. Reindexing to ensure a regular time series.
        3. Flagging cut-in speed for the turbine.
        4. Engineering time-based features (week, month, hour, season, day/night).
        5. Merging external temperature data via meteostat.
        6. Tracking imputed rows, shifting target by one step.
        7. Generating delta (change) features.
        8. One-hot encoding categorical features.

    Returns:
        pd.DataFrame: A preprocessed DataFrame ready for modeling.
    """
    logger.info(f"Reading data file: {s.DATA_FILE}")
    df = pd.read_csv(
        s.DATA_FILE,
        names=["datetime", "active_power", "wind_speed", "theoretical_power", "wind_direction"],
        skiprows=1
    )
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df = df.astype(float)

    # Reindex to a full time range at the given frequency
    logger.info(f"Reindexing data to frequency {s.FREQ}")
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=s.FREQ)
    df = df.reindex(full_range)
    df.index.name = "datetime"

    # Determine cut-in speed
    logger.info("Determining cut-in speed from theoretical power > 0 observations.")
    cut_in_speed = df[df["theoretical_power"] > 0]["wind_speed"].min()
    cut_in_speed = round(cut_in_speed, 1)

    # Create binary flags
    df["cut_speed"] = df["wind_speed"] > cut_in_speed
    df["power_flag"] = df["active_power"] > 0

    # Time-based features
    df["week"] = df.index.isocalendar().week
    df["month"] = df.index.month
    df["hour"] = df.index.hour
    df["season"] = df["month"].apply(get_season)

    # Day/Night feature
    logger.info("Adding day/night feature based on Astral sun calculations.")
    loc = LocationInfo(s.LOCATION_NAME, s.REGION_NAME, s.TIMEZONE, s.LAT, s.LON)
    df["is_night"] = df.index.map(lambda ts: is_day_or_night(ts, loc))

    # Merge external temperature (meteostat)
    logger.info("Fetching and merging weather data using Meteostat.")
    meteo_location = Point(s.LAT, s.LON)
    start, end = df.index.min(), df.index.max()
    data_hourly = Hourly(meteo_location, start, end).fetch()
    data_10min = data_hourly.resample(s.FREQ).ffill().reset_index()
    weather_df = data_10min[["time", "temp"]].rename(columns={"time": "datetime"})
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
    weather_df.set_index("datetime", inplace=True)
    df = pd.merge(df, weather_df, left_index=True, right_index=True, how="left")
    df["temp"] = df["temp"].interpolate()

    # Track rows where any column was missing
    mask = df.isna().any(axis=1).astype(int)
    df = df.fillna(method="ffill")
    df["imputation_mask"] = mask

    # Shift active_power (forecasting next step)
    df["active_power"] = df["active_power"].shift(1)

    # Delta features
    df["delta_wind_direction"] = df["wind_direction"].shift(1) - df["wind_direction"].shift(2)
    df["delta_wind_speed"] = df["wind_speed"].shift(1) - df["wind_speed"].shift(2)
    df["delta_temp"] = df["temp"].shift(1) - df["temp"].shift(2)
    df["delta_active_power"] = df["active_power"].shift(1) - df["active_power"].shift(2)

    # Drop remaining NaNs
    logger.info("Dropping remaining NaN values after shifting and feature engineering.")
    df.dropna(inplace=True)

    # One-hot encode
    logger.info("Performing one-hot encoding on season and other categorical features.")
    df = pd.get_dummies(df, columns=["season"])
    df = pd.get_dummies(df)

    logger.info("Data preprocessing complete. Returning DataFrame.")
    return df

def get_season(month):
    """
    Maps a month integer to its corresponding season.

    Args:
        month (int): Numeric month (1-12).

    Returns:
        str: Season name ("winter", "spring", "summer", or "autumn").
    """
    return {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer"
    }.get(month, "autumn")

def is_day_or_night(dt, location):
    """
    Determines whether a given timezone-aware datetime is daytime or nighttime.

    Args:
        dt (pd.Timestamp): Timezone-naive timestamp to localize.
        location (astral.LocationInfo): Astral location information used to calculate sunrise/sunset.

    Returns:
        int: 0 if it's daytime, 1 if it's nighttime.
    """
    dt_local = dt.tz_localize(s.TIMEZONE)
    s_obj = sun(location.observer, date=dt_local)
    return 0 if s_obj["sunrise"] < dt_local < s_obj["sunset"] else 1
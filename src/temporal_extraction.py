import pandas as pd
import numpy as np

def process_datetime_and_trigonometric_features(df):
    df = df.copy()
    df = df[df['startTime'] != -1]
    df = df[df['stripeSize'] != -1]

    df['startTime'] = pd.to_datetime(df['startTime'])

    reference_time = df['startTime'].iloc[0]
    df['relativeStartTime'] = (df['startTime'] - reference_time).dt.total_seconds()

    # Month-day cycle
    day_of_year = df['startTime'].dt.dayofyear
    df['startMonthDaySin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['startMonthDayCos'] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Time-of-day cycle
    seconds = (df['startTime'].dt.hour * 3600 +
               df['startTime'].dt.minute * 60 +
               df['startTime'].dt.second)
    df['startDaytimeSin'] = np.sin(2 * np.pi * seconds / 86400)
    df['startDaytimeCos'] = np.cos(2 * np.pi * seconds / 86400)

    return df

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.utils.constants import SENSOR_COLS, SETTING_COLS, CONSTANT_SENSORS_FD001

def add_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """
    Adds RUL column to the dataframe.
    RUL is clipped at max_rul (Piecewise Linear RUL).
    """
    # Calculate RUL: Max Time - Current Time
    # First, get max time per unit
    max_time = df.groupby('unit')['time'].max().reset_index()
    max_time.columns = ['unit', 'max_time']
    
    df = df.merge(max_time, on='unit', how='left')
    df['RUL'] = df['max_time'] - df['time']
    df.drop('max_time', axis=1, inplace=True)
    
    # Clip RUL
    df['RUL'] = df['RUL'].clip(upper=max_rul)
    return df

def process_data(train_df, test_df, subset="FD001"):
    """
    Process data:
    1. Drop constant sensors.
    2. Normalize sensors (fit on train, transform test).
    """
    sensors_to_drop = CONSTANT_SENSORS_FD001 if subset == "FD001" else []
    
    # Identify useful sensors
    useful_sensors = [s for s in SENSOR_COLS if s not in sensors_to_drop]
    # We might also want to include settings? usually settings are features too.
    # For FD001, settings are mostly constant or simple. Let's stick to sensors first or include all.
    # Detailed papers say for FD001, settings are also not very useful or constant. 
    # Let's use useful_sensors + settings if they vary.
    
    # Feature columns
    feature_cols = [c for c in useful_sensors] 
    # (ignoring settings for now as per common FD001 practice, or check variance)
    
    scaler = MinMaxScaler()
    
    # Fit scaler on Train
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    
    # Transform Test
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_df, test_df, feature_cols

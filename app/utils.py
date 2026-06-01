import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

# Define rainfall classification thresholds
# Class mapping:
# 0: No Rain (< 0.1 mm)
# 1: Light Rain (0.1 - 2.5 mm)
# 2: Moderate Rain (2.5 - 10.0 mm)
# 3: Heavy Rain (>= 10.0 mm)
RAINFALL_CLASSES = ["No Rain", "Light Rain", "Moderate Rain", "Heavy Rain"]

def classify_rainfall(val: float) -> int:
    """
    Classifies a continuous rainfall amount (mm) into categorical bins:
    0: No Rain, 1: Light Rain, 2: Moderate Rain, 3: Heavy Rain.
    """
    if pd.isna(val) or val < 0.1:
        return 0
    elif val < 2.5:
        return 1
    elif val < 10.0:
        return 2
    else:
        return 3

def get_class_name(class_idx: int) -> str:
    """Returns the descriptive string of a rainfall category."""
    if 0 <= class_idx < len(RAINFALL_CLASSES):
        return RAINFALL_CLASSES[class_idx]
    return "Unknown"

def handle_outliers(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Clips extreme outliers for continuous atmospheric features using the IQR method.
    Note: Rain is excluded from traditional IQR clipping to preserve true heavy precipitation events,
    which are statistically highly skewed but physically valid.
    """
    df_clean = df.copy()
    for col in columns:
        if col not in df_clean.columns or col == "Rainfall":
            continue
        
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Clip values to bounds
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Cap rainfall at an extreme physical threshold (e.g., 500mm/day) if any non-physical value exists
    if "Rainfall" in df_clean.columns:
        df_clean["Rainfall"] = df_clean["Rainfall"].clip(lower=0, upper=500.0)
        
    return df_clean

def preprocess_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning, imputation, and feature engineering.
    Engineers:
    - Calendar sine/cosine cyclical features (Month, Day of Year)
    - Lag variables (1, 2, and 3-day lags)
    - Rolling means (3-day and 7-day windows)
    - Rainfall classifications
    - Drought & flood risk markers
    """
    df_feat = df.copy()
    
    # Ensure datetime parsing and sorting
    df_feat["Date"] = pd.to_datetime(df_feat["Date"])
    df_feat = df_feat.sort_values(by=["City", "Date"]).reset_index(drop=True)
    
    # Impute missing values (using forward/backward fill grouped by city)
    for col in ["Temperature", "Humidity", "Pressure", "Wind_Speed", "Rainfall"]:
        df_feat[col] = df_feat.groupby("City")[col].transform(lambda x: x.ffill().bfill())
    
    # Clip outliers for numerical weather variables
    continuous_cols = ["Temperature", "Humidity", "Pressure", "Wind_Speed"]
    df_feat = handle_outliers(df_feat, continuous_cols)
    
    # Sine/Cosine Month Features for cyclical time representation
    df_feat["Month_Sin"] = np.sin(2 * np.pi * df_feat["Month"] / 12.0)
    df_feat["Month_Cos"] = np.cos(2 * np.pi * df_feat["Month"] / 12.0)
    
    # Target classification mapping
    df_feat["Rainfall_Class"] = df_feat["Rainfall"].apply(classify_rainfall)
    
    # Create City-grouped Lag and Rolling Features to prevent leakage across cities
    engineered_dfs = []
    for city, group in df_feat.groupby("City"):
        group = group.copy().sort_values(by="Date")
        
        # 1, 2, 3 day Lags for key features
        for lag in [1, 2, 3]:
            for col in ["Rainfall", "Temperature", "Humidity", "Wind_Speed"]:
                group[f"{col}_Lag_{lag}"] = group[col].shift(lag)
        
        # Rolling means
        for window in [3, 7]:
            for col in ["Temperature", "Humidity", "Pressure"]:
                group[f"{col}_RollMean_{window}"] = group[col].rolling(window=window).mean()
        
        # Risk Calculations
        # 1. Consecutive dry days (Rainfall < 0.1 mm)
        rain_mask = group["Rainfall"] < 0.1
        group["Dry_Spell_Days"] = rain_mask.groupby((~rain_mask).cumsum()).cumsum()
        
        # 2. Cumulative 30-day precipitation deficit
        group["Rain_30d_Sum"] = group["Rainfall"].rolling(window=30, min_periods=1).sum()
        
        # Calculate historical median for that month to flag departures
        month_medians = group.groupby("Month")["Rain_30d_Sum"].transform("median")
        group["Precip_Deficit"] = month_medians - group["Rain_30d_Sum"]
        
        # Drought Index Warning: 30-day rain is less than 30% of the historical median for that month AND dry spell is > 15 days
        group["Drought_Risk"] = ((group["Rain_30d_Sum"] < 0.3 * month_medians) & (group["Dry_Spell_Days"] > 15)).astype(int)
        
        # 3. Flood/Extreme precipitation warning: daily rainfall > 50 mm
        group["Flood_Risk"] = (group["Rainfall"] > 50.0).astype(int)
        
        engineered_dfs.append(group)
        
    df_final = pd.concat(engineered_dfs, ignore_index=True)
    
    # Drop rows with NaN from lag/rolling operations (first 7 days of each city)
    df_final = df_final.dropna().reset_index(drop=True)
    return df_final

def prepare_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Rainfall",
    lookback_window: int = 60,
    scaler_x: MinMaxScaler = None,
    scaler_y: MinMaxScaler = None
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Generates 3D sequences for LSTM training: (samples, lookback_window, features).
    Fits scalers if not provided, otherwise applies them.
    Returns:
    - X_seq: 3D array of scaled feature sequences
    - y_seq: 1D array of scaled target values (next-day rainfall)
    - scaler_x: Scaler used for features
    - scaler_y: Scaler used for targets
    """
    df_sorted = df.sort_values(by=["City", "Date"]).reset_index(drop=True)
    
    if scaler_x is None:
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        df_sorted[feature_cols] = scaler_x.fit_transform(df_sorted[feature_cols])
    else:
        df_sorted[feature_cols] = scaler_x.transform(df_sorted[feature_cols])
        
    if scaler_y is None:
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        # Keep fit shape 2D
        df_sorted[[target_col]] = scaler_y.fit_transform(df_sorted[[target_col]])
    else:
        df_sorted[[target_col]] = scaler_y.transform(df_sorted[[target_col]])
        
    X_list, y_list = [], []
    
    # Create sequence indices per city to avoid bleeding cross-city historical records
    for _, city_df in df_sorted.groupby("City"):
        feat_data = city_df[feature_cols].values
        target_data = city_df[target_col].values
        
        for i in range(lookback_window, len(city_df)):
            X_list.append(feat_data[i - lookback_window : i])
            y_list.append(target_data[i])
            
    return np.array(X_list), np.array(y_list), scaler_x, scaler_y

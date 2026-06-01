#!/usr/bin/env python3
"""
Rainfall Prediction and Climate Trend Analysis using LSTM - Data Downloader
Downloads historical daily weather observations from the NASA POWER API (2011-2025)
for multiple climatically diverse cities: Mumbai, New Delhi, Bengaluru, Chennai, and Kolkata.
"""

import os
import time
import logging
from typing import Dict, Any, List
import pandas as pd
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of target stations/cities with geographical coordinates
CITIES: List[Dict[str, Any]] = [
    {"name": "Mumbai", "latitude": 19.0760, "longitude": 72.8777},
    {"name": "New Delhi", "latitude": 28.6139, "longitude": 77.2090},
    {"name": "Bengaluru", "latitude": 12.9716, "longitude": 77.5946},
    {"name": "Chennai", "latitude": 13.0827, "longitude": 80.2707},
    {"name": "Kolkata", "latitude": 22.5726, "longitude": 88.3639}
]

# NASA POWER API daily temporal query details
# Variables of interest:
# - PRECTOTCORR: Precipitation Corrected (mm/day)
# - T2M: Temperature at 2 Meters (°C)
# - RH2M: Relative Humidity at 2 Meters (%)
# - PS: Surface Pressure (kPa)
# - WS2M: Wind Speed at 2 Meters (m/s)
PARAMETERS = "PRECTOTCORR,T2M,RH2M,PS,WS2M"
START_DATE = "20110101"
END_DATE = "20251231"
API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

def download_city_weather(
    city_name: str, 
    lat: float, 
    lon: float, 
    retries: int = 3, 
    backoff: int = 5
) -> pd.DataFrame:
    """
    Downloads weather observations from NASA POWER API for a given coordinate.
    """
    params = {
        "parameters": PARAMETERS,
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON"
    }
    
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading data for {city_name} (Lat: {lat}, Lon: {lon}) - Attempt {attempt}/{retries}")
            response = requests.get(API_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                features = data.get("properties", {}).get("parameter", {})
                
                # Check that we got all requested variables
                if not features:
                    logger.warning(f"No properties.parameter found in response for {city_name}.")
                    continue
                
                # Form dataframe
                df_dict = {"Date": [], "City": [], "Latitude": [], "Longitude": []}
                for var in ["PRECTOTCORR", "T2M", "RH2M", "PS", "WS2M"]:
                    df_dict[var] = []
                
                # Extract timestamps
                dates = list(features.get("T2M", {}).keys())
                if not dates:
                    logger.warning(f"No dates found in weather variables for {city_name}")
                    continue
                
                for date_str in dates:
                    # Date format from NASA is YYYYMMDD
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                    df_dict["Date"].append(formatted_date)
                    df_dict["City"].append(city_name)
                    df_dict["Latitude"].append(lat)
                    df_dict["Longitude"].append(lon)
                    
                    for var in ["PRECTOTCORR", "T2M", "RH2M", "PS", "WS2M"]:
                        val = features.get(var, {}).get(date_str, None)
                        # NASA API returns -999 for missing values
                        if val == -999.0 or val is None:
                            df_dict[var].append(None)
                        else:
                            df_dict[var].append(val)
                
                df = pd.DataFrame(df_dict)
                logger.info(f"Successfully processed {len(df)} records for {city_name}")
                return df
            
            else:
                logger.warning(f"Failed response: HTTP {response.status_code} for {city_name}")
                
        except Exception as e:
            logger.error(f"Error during download for {city_name}: {e}")
            
        if attempt < retries:
            logger.info(f"Sleeping for {backoff} seconds before retry...")
            time.sleep(backoff)
            backoff *= 2
            
    raise RuntimeError(f"Could not retrieve data for {city_name} after {retries} retries.")

def main() -> None:
    # Ensure raw data folder exists
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    logger.info("Initializing NASA POWER weather data download pipeline...")
    logger.info(f"Retrieving observations from {START_DATE} to {END_DATE}")
    
    combined_dfs = []
    
    for city in CITIES:
        try:
            city_df = download_city_weather(
                city_name=city["name"], 
                lat=city["latitude"], 
                lon=city["longitude"]
            )
            combined_dfs.append(city_df)
            # NASA POWER API asks for polite rate limiting
            time.sleep(2)
        except Exception as e:
            logger.critical(f"Aborting download pipeline due to failure at {city['name']}: {e}")
            return
            
    if combined_dfs:
        full_df = pd.concat(combined_dfs, ignore_index=True)
        
        # Format Column Names for Better Readability
        full_df.rename(columns={
            "PRECTOTCORR": "Rainfall",
            "T2M": "Temperature",
            "RH2M": "Humidity",
            "PS": "Pressure",
            "WS2M": "Wind_Speed"
        }, inplace=True)
        
        # Convert date and extract year/month
        full_df["Date"] = pd.to_datetime(full_df["Date"])
        full_df["Month"] = full_df["Date"].dt.month
        full_df["Year"] = full_df["Date"].dt.year
        
        # Save output
        output_path = os.path.join(raw_dir, "weather_data.csv")
        full_df.to_csv(output_path, index=False)
        logger.info(f"Successfully compiled all weather observations! Saved to: {output_path}")
        logger.info(f"Total Rows: {len(full_df)}")
        logger.info(f"Memory Usage: {full_df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        
        # Basic inspection output
        logger.info("\nData Overview:\n" + str(full_df.head(3)))
        logger.info("\nData Summary Statistics:\n" + str(full_df.describe().T))
    else:
        logger.error("No data retrieved.")

if __name__ == "__main__":
    main()

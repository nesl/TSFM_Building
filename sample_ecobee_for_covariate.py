#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:54:29 2023

@author: ozanbaris
"""
import xarray as xr
import pandas as pd
import numpy as np
import os
from geopy.exc import GeocoderServiceError, GeocoderUnavailable
from geopy.geocoders import Nominatim
from math import sqrt
import time

# File names
file_names = ['Jun_clean.nc', 'Jul_clean.nc', 'Aug_clean.nc']

# Initialize an empty DataFrame
df = pd.DataFrame()

# Load each file and append it to df
for file_name in file_names:
    data = xr.open_dataset(file_name)
    temp_df = data.to_dataframe()
    temp_df = temp_df.reset_index()
    df = pd.concat([df, temp_df], ignore_index=True)
#%%

def resample_data(df, sensor_count, columns_to_average, columns_to_sum, columns_to_take_first, motion_cols):
    """
    Resample data to a 60-minute resolution.

    Parameters:
    - df: DataFrame containing the data to be resampled.
    - sensor_count: Number of sensors, used to determine motion columns.
    - columns_to_average: List of column names whose values should be averaged during resampling.
    - columns_to_sum: List of column names whose values should be summed during resampling.
    - columns_to_take_first: List of column names for which the first value should be taken during resampling.
    - motion_cols: List of column names for motion detection data.

    Returns:
    - DataFrame with resampled data.
    """

    # Initialize an empty DataFrame to store the final resampled data
    resampled_df = pd.DataFrame()
    
    # Step 1: Replace NaN values with zeros in CoolingEquipmentStage1_RunTime, CoolingEquipmentStage2_RunTime, and CoolingEquipmentStage3_RunTime
    df['CoolingEquipmentStage1_RunTime'].fillna(0, inplace=True)
    df['CoolingEquipmentStage2_RunTime'].fillna(0, inplace=True)
    df[motion_cols].fillna(0, inplace=True)

    # Step 2: Summing up values of CoolingEquipmentStage1_RunTime and CoolingEquipmentStage2_RunTime
    df['CoolingRunTime'] = df['CoolingEquipmentStage1_RunTime'] + df['CoolingEquipmentStage2_RunTime'] 
    # Loop over each unique house ID to perform resampling
    # Step 1: Replace NaN values with zeros in CoolingEquipmentStage1_RunTime, CoolingEquipmentStage2_RunTime, and CoolingEquipmentStage3_RunTime
    df['HeatingEquipmentStage1_RunTime'].fillna(0, inplace=True)
    df['HeatingEquipmentStage2_RunTime'].fillna(0, inplace=True)
    df[motion_cols].fillna(0, inplace=True)

    # Step 2: Summing up values of CoolingEquipmentStage1_RunTime and CoolingEquipmentStage2_RunTime
    df['HeatingRunTime'] = df['HeatingEquipmentStage1_RunTime'] + df['HeatingEquipmentStage2_RunTime'] 
    for house_id in df['id'].unique():
        house_data = df[df['id'] == house_id]

        # Perform resampling for each type of column
        avg_resampled = house_data.resample('60T', on='time')[columns_to_average].mean().reset_index()
        sum_resampled = house_data.resample('60T', on='time')[columns_to_sum].sum().reset_index()
        first_resampled = house_data.resample('60T', on='time')[columns_to_take_first].first().reset_index()
        max_resampled = house_data.resample('60T', on='time')[motion_cols].max().reset_index()  # Use max for motion columns

        # Merge all the resampled dataframes for this house
        house_resampled_df = avg_resampled.merge(sum_resampled, on='time').merge(first_resampled, on='time').merge(max_resampled, on='time')

        # Append this house's resampled data to the final dataframe
        resampled_df = pd.concat([resampled_df, house_resampled_df], ignore_index=True)

    # Ensure that the 'id' column is correctly merged after all operations
    resampled_df['id'] = resampled_df['id'].fillna(method='ffill')

    return resampled_df

sensor_count = 5
remote_sensor_cols = [
    'RemoteSensor1_Temperature',
    'RemoteSensor2_Temperature',
    'RemoteSensor3_Temperature',
    'RemoteSensor4_Temperature',
    'RemoteSensor5_Temperature'
]

motion_cols = ['Thermostat_DetectedMotion'] + [f'RemoteSensor{i}_DetectedMotion' for i in range(1, sensor_count + 1)]
columns_to_average = ['Indoor_AverageTemperature', 'Thermostat_Temperature', 'Outdoor_Temperature'] + remote_sensor_cols
columns_to_sum = ['CoolingRunTime', 'HeatingRunTime']
columns_to_take_first = ['id', 'State']

# Resample the DataFrame
resampled_df = resample_data(df, sensor_count, columns_to_average, columns_to_sum, columns_to_take_first, motion_cols)

print("Resampled DataFrame:", resampled_df.head())



#%%
def has_continuous_data(df, columns, days_required=32):
    combined_series = df[columns].notna().all(axis=1).astype(int)
    continuous_count = combined_series.rolling(window=days_required*24).sum()
    return (continuous_count == days_required*24).any()

# Dictionaries for full datasets
one_houses = {}
two_houses = {}
three_houses = {}
four_houses = {}
five_houses = {}

unique_house_ids = resampled_df['id'].unique()
for house_id in unique_house_ids:
    house_df = resampled_df[resampled_df['id'] == house_id].copy()
    house_df.sort_values(by="time", inplace=True)
    
    # Identify sensors with 16 days of continuous data for the house
    valid_sensors_for_house = [col for col in remote_sensor_cols if has_continuous_data(house_df, [col])]
    sensor_counts = len(valid_sensors_for_house)
    
    if sensor_counts == 1:
        one_houses[house_id] = house_df
    elif sensor_counts == 2:
        two_houses[house_id] = house_df
    elif sensor_counts == 3:
        three_houses[house_id] = house_df
    elif sensor_counts == 4:
        four_houses[house_id] = house_df
    elif sensor_counts == 5:
        five_houses[house_id] = house_df

all_houses_classified = {
    1: one_houses,
    2: two_houses,
    3: three_houses,
    4: four_houses,
    5: five_houses
}


 #%%

# Initialize dictionaries for each sensor count in all_houses_reduced
one_houses_reduced = {}
two_houses_reduced = {}
three_houses_reduced = {}
four_houses_reduced = {}
five_houses_reduced = {}

all_houses_reduced = {
    1: one_houses_reduced,
    2: two_houses_reduced,
    3: three_houses_reduced,
    4: four_houses_reduced,
    5: five_houses_reduced
}

# Iterate over each sensor count and the corresponding houses
for sensor_count, houses in all_houses_classified.items():
    for house_id, house_df in houses.items():
        # Define columns for sensor data and relevant conditions
        current_remote_sensor_cols = ['RemoteSensor{}_Temperature'.format(i) for i in range(1, sensor_count + 1)]
        relevant_cols = ['Indoor_AverageTemperature', 'Thermostat_Temperature', 'Outdoor_Temperature', 
                         'CoolingRunTime', 'HeatingRunTime', 'time'] + current_remote_sensor_cols

        # Filter out rows where any relevant column has NaN values
        house_df = house_df.dropna(subset=relevant_cols)

        # Ensure the dataframe still contains data after filtering
        if house_df.empty:
            print(f"House {house_id} has no valid data after filtering NaNs.")
            continue

        # Sort data by time to ensure continuous intervals are properly checked
        house_df.sort_values('time', inplace=True)

        # Attempt to find a 32-day period where CoolingRunTime has some non-zero values and HeatingRunTime is always zero
        # Calculate the number of hours in 32 days assuming hourly data
        total_hours = 32 * 24
        for start_idx in range(len(house_df) - total_hours + 1):
            subset = house_df.iloc[start_idx: start_idx + total_hours]
            start_time = subset['time'].iloc[0]
            end_time = subset['time'].iloc[-1]
            delta_days = (end_time - start_time).days

            # Check if the subset covers exactly 32 days and meets other criteria
            if delta_days == 31 and subset['CoolingRunTime'].gt(0).any(): #and subset['HeatingRunTime'].eq(0).all():
                all_houses_reduced[sensor_count][house_id] = subset
                print(f"Found a valid 32-day period for house {house_id} with sensor count {sensor_count}.")
                break  # Stop after finding the first valid 32-day period

            if start_idx + total_hours == len(house_df):  # Check if this is the last possible window
                print(f"No valid 32-day period found for house {house_id} with sensor count {sensor_count}.")

 

#%%
def validate_house_data(house_data, sensor_cols):
    for house_id, house_df in house_data.items():
        nan_cols = [col for col in sensor_cols if house_df[col].isna().any()]
        if nan_cols:
            print(f"House {house_id} has NaN values in columns: {', '.join(nan_cols)}")
            for col in nan_cols:
                print(house_df[house_df[col].isna()][col])


# For the sake of validation, iterate over all the reduced house data dictionaries:
for sensor_count in range(1, 6):
    current_remote_sensor_cols = ['RemoteSensor{}_Temperature'.format(i) for i in range(1, sensor_count + 1)]
    validate_house_data(all_houses_reduced[sensor_count], current_remote_sensor_cols)
    print(f"Number of houses with 1 additional sensor: {len(all_houses_reduced[sensor_count])}")



#%%    

ca_data = pd.read_csv('CA.csv')
il_data = pd.read_csv('IL.csv')
ny_data = pd.read_csv('NY.csv')
tx_data = pd.read_csv('TX.csv')
#%% 
remote_sensor_cols = [
    'RemoteSensor1_Temperature', 
    'RemoteSensor2_Temperature', 
    'RemoteSensor3_Temperature', 
    'RemoteSensor4_Temperature', 
    'RemoteSensor5_Temperature'
]

state_dict = {}

house_dictionaries = [one_houses, two_houses, three_houses, four_houses, five_houses]

for sensor_count, house_dict in all_houses_reduced.items():
    for house_id, single_house_data in house_dict.items():
        state_values = single_house_data['State'].dropna().unique()
        assigned_state = 'Unknown'
        for state in state_values:
            if state != '':
                assigned_state = state
                break
        state_dict[house_id] = assigned_state


data_map = {
    'CA': ca_data,
    'TX': tx_data,
    'IL': il_data,
    'NY': ny_data
}

# Assign city to each house in house dictionaries based on the state and house_id
for house_id, state in state_dict.items():
    if state in data_map:
        house_row = data_map[state][data_map[state]['Identifier'] == house_id]
        if not house_row.empty:
            city = house_row['City'].iloc[0]

            # Find the dictionary where this house belongs and update the city
            for sensor_count, house_dict in all_houses_reduced.items():
                if house_id in house_dict:
                    if 'City' not in house_dict[house_id]:
                        house_dict[house_id]['City'] = city
                    else:
                        house_dict[house_id]['City'] = city
                    print(city)
                    break


def get_lat_lon(city, state, retries=3):
    geolocator = Nominatim(user_agent="yourUniqueAppNameHere")
    while retries > 0:
        try:
            location = geolocator.geocode(f"{city}, {state}")
            return (location.latitude, location.longitude)
        except (GeocoderServiceError, GeocoderUnavailable):
            retries -= 1
            time.sleep(2)  # Wait for 2 seconds before retrying
    raise Exception("Failed to get coordinates after multiple retries")

#%%    


def read_solar_data(base_folder):
    """
    Read solar data from CSV files in the specified folder structure.
    Solar data is extracted from NATIONAL SOLAR DATABASE API for the lat and lon values of the city, state tuples. 
    """
    lat_lon_data_dict = {}  # Dictionary to hold data, keyed by (lat, lon)
    
    # Iterate through folders and files
    for foldername in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, foldername)
        
        print(f"Checking folder: {foldername}")
        
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                
                if filename.endswith('.csv'):
                    file_path = os.path.join(folder_path, filename)
                    
                    print(f"Reading file: {filename}")
                    
                    # Read only the first row to get metadata
                    metadata_df = pd.read_csv(file_path, nrows=1)
                    lat = metadata_df.at[0, 'Latitude']
                    lon = metadata_df.at[0, 'Longitude']
                    
                    print(f"Latitude: {lat}, Longitude: {lon}")
                    
                    # Read the rest of the file skipping the first two rows to get actual data
                    df = pd.read_csv(file_path, skiprows=2)
                    
                    # Validate that the required columns exist before subsetting the dataframe
                    if all(col in df.columns for col in ['Year', 'Month', 'Day', 'Hour', 'Minute', 'GHI', 'DNI']):
                        # Filtering rows based on the 'Month' column
                        df = df[df['Month'].isin([6, 7, 8])]
                        
                        df = df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'GHI', 'DNI']]
                        # Use lat and lon as a key to store this dataframe
                        lat_lon_data_dict[(lat, lon)] = df
                        print(f"Data saved for coordinates ({lat}, {lon})")
                    else:
                        print(f"Required columns not found in file {filename}")

    return lat_lon_data_dict

base_folder = 'solar_data'  # Replace with the path to your solar_data folder
lat_lon_data_dict = read_solar_data(base_folder)
    
print("Reading completed. Data dictionary keys (lat, lon):")
print(lat_lon_data_dict.keys())
#%% 

def get_lat_lon(city, state, retries=3):
    geolocator = Nominatim(user_agent="yourUniqueAppNameHere")
    while retries > 0:
        try:
            location = geolocator.geocode(f"{city}, {state}")
            return (location.latitude, location.longitude)
        except (GeocoderServiceError, GeocoderUnavailable):
            retries -= 1
            time.sleep(2)
    raise Exception("Failed to get coordinates after multiple retries")

def find_nearest_station(city_latitude, city_longitude, lat_lon_data_dict_keys):
    min_distance = float('inf')
    nearest_station = None
    
    for station_latitude, station_longitude in lat_lon_data_dict_keys:
        distance = sqrt((station_latitude - city_latitude)**2 + (station_longitude - city_longitude)**2)
        
        if distance < min_distance:
            min_distance = distance
            nearest_station = (station_latitude, station_longitude)
            
    return nearest_station

# Define a function to combine date and time columns into a single datetime column
def combine_datetime(df):
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']]
                                    .astype(str).apply(lambda x: ' '.join(x), axis=1),
                                    format='%Y %m %d %H %M')

# Your existing loop
for house_group, houses_dict in all_houses_reduced.items():
    for house_id, house_data in houses_dict.items():

        # Get city name and state from the data and state_dict respectively
        city_name = house_data['City'].mode().iloc[0]
        state_name = state_dict[house_id]
        
        try:
            # Fetch latitude and longitude using get_lat_lon function
            latitude, longitude = get_lat_lon(city_name, state_name)
            
            # Find the nearest station
            nearest_station = find_nearest_station(latitude, longitude, lat_lon_data_dict.keys())

            try:
                solar_data = lat_lon_data_dict[nearest_station].copy()

                # Combine Year, Month, Day, Hour, Minute columns into a single datetime column for solar_data
                combine_datetime(solar_data)

                # Convert 'time' column in house_data into a datetime column if it's not
                house_data.loc[:, 'time'] = pd.to_datetime(house_data['time'])


                # Merge house_data with solar_data based on the time
                merged_data = pd.merge(house_data, solar_data, how='left', left_on='time', right_on='datetime')

                # Update house_data with the merged data
                houses_dict[house_id] = merged_data

                #print(f"Solar data for house_id {house_id} in {city_name}, {state_name} has been merged.")
                
            except KeyError:
                print(f"Solar data for house_id {house_id} in {city_name}, {state_name}: No data available")
                print(f"Keys available in lat_lon_data_dict: {list(lat_lon_data_dict.keys())}")
                
        except Exception as e:
            print(f"Failed to fetch coordinates for {city_name}, {state_name}: {e}")




# Create a main directory to store all the subfolders
main_output_directory = "house_data_csvs0"
if not os.path.exists(main_output_directory):
    os.makedirs(main_output_directory)

# Function to save each DataFrame to a CSV file in separate folders
def save_dfs_to_csv(all_houses_dict, main_output_directory):
    for house_group, houses_dict in all_houses_reduced.items():
        
        # Create a subdirectory for each house_group
        sub_output_directory = os.path.join(main_output_directory, f"house_group_{house_group}")
        if not os.path.exists(sub_output_directory):
            os.makedirs(sub_output_directory)
        
        for house_id, house_data in houses_dict.items():
            csv_filename = os.path.join(sub_output_directory, f"house_id_{house_id}.csv")
            house_data.to_csv(csv_filename, index=False)

save_dfs_to_csv(all_houses_reduced, main_output_directory)
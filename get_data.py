
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import os
from datasets import load_dataset
from datasets import load_dataset
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def get_real_building_data(duration, pred_hrz, sampling_rate, mode, occupancy='occupied', batch_id=0, batch_number=10):
    # Read the data
    temp_data = pd.read_csv('temp_data.csv')

    # Combine 'Date' and 'Time' columns and ensure they are in the correct format
    temp_data['Timestamp'] = pd.to_datetime(temp_data['Date'] + ' ' + temp_data['Time'], format='%Y-%m-%d 0 days %H:%M:%S', errors='coerce')

    # Drop rows where 'Timestamp' could not be parsed
    temp_data = temp_data.dropna(subset=['Timestamp'])

    # Sort the data by Timestamp to ensure the sequence is correct
    temp_data = temp_data.sort_values(by='Timestamp')

    # Ensure unique timestamps after parsing
    temp_data = temp_data.drop_duplicates(subset=['Timestamp'])

    # Identify the longest sequence of consecutive zeros in 'Fan (sec)'
    temp_data['is_zero'] = temp_data['Fan (sec)'] == 0
    temp_data['group'] = (temp_data['is_zero'] != temp_data['is_zero'].shift()).cumsum()

    # Filter groups where 'Fan (sec)' is zero and calculate the size of each group
    zero_groups = temp_data[temp_data['is_zero']].groupby('group').size()

    # Find the group with the longest sequence of zeros
    longest_zero_group = zero_groups.idxmax()
    longest_zero_duration = zero_groups.max()

    # Extract the data corresponding to the longest sequence of zeros
    longest_zero_data = temp_data[temp_data['group'] == longest_zero_group]

    # Drop the auxiliary columns used for calculations
    longest_zero_data = longest_zero_data.drop(columns=['is_zero', 'group'])

    #print the beginning and end of longest_zero_data
    #print(longest_zero_data.head())
    #print(longest_zero_data.tail())
    #print the duration of longest_zero_data in hours
    #print((longest_zero_data['Timestamp'].iloc[-1] - longest_zero_data['Timestamp'].iloc[0]).days)
    # Ensure the 'Date' column is in datetime format
    temp_data['Date'] = pd.to_datetime(temp_data['Date'], errors='coerce')

    if mode == 'off':
        dataset = longest_zero_data
    elif mode == 'heat' and occupancy == 'occupied':
        filtered_data = temp_data[(temp_data['Date'] >= pd.to_datetime('2023-11-02')) & (temp_data['Date'] < pd.to_datetime('2023-12-16'))]
        dataset = filtered_data
    elif mode == 'heat' and occupancy == 'unoccupied':
        filtered_data = temp_data[(temp_data['Date'] >= pd.to_datetime('2023-12-17')) & (temp_data['Date'] < pd.to_datetime('2024-01-02'))]
        dataset = filtered_data

    len_gt = int(pred_hrz * 3600 / sampling_rate)
    len_data = int(duration * 3600 / sampling_rate)

    # Resample only numeric columns
    numeric_columns = ['Thermostat Temperature (F)', 'Outdoor Temp (F)', 'Fan (sec)', 'Heat Set Temp (F)']
    if sampling_rate != 300:
        dataset_resampled = dataset.set_index('Timestamp')[numeric_columns].resample(f'{sampling_rate}S').mean().reset_index()
    else:
        dataset_resampled = dataset[numeric_columns + ['Timestamp']]

    # Interpolate the nan values in the dataset
    dataset_resampled = dataset_resampled.interpolate()

    # Ensure no duplicates after resampling
    dataset_resampled = dataset_resampled.drop_duplicates(subset=['Timestamp'])

    # Calculate batch size in terms of number of samples
    total_size = len(dataset_resampled)
    max_start_point = total_size - (len_data + len_gt)
    interval = max_start_point // (batch_number - 1)

    # Create start points
    start_points = [i * interval for i in range(batch_number)]
    start_points[-1] = max_start_point

    data_start = start_points[batch_id]
    data_end = data_start + len_data
    test_data_start = data_end
    test_data_end = test_data_start + len_gt

    # Extract the required columns and timestamp for data and test_data within the batch
    data = dataset_resampled[['Thermostat Temperature (F)', 'Outdoor Temp (F)', 'Fan (sec)', 'Heat Set Temp (F)', 'Timestamp']].values[data_start:data_end]
    test_data = dataset_resampled[['Thermostat Temperature (F)', 'Outdoor Temp (F)', 'Fan (sec)', 'Heat Set Temp (F)', 'Timestamp']].values[test_data_start:test_data_end]

    return data, test_data
def get_electricity_data(duration, pred_hrz, sampling_rate, occupancy, batch_id=0, batch_number=10):
    # Read the data
    sense_data = pd.read_csv('sense_data.csv')
    
    # Ensure 'Timestamp' column is in datetime format
    sense_data['Timestamp'] = pd.to_datetime(sense_data['Timestamp'], errors='coerce')

    # Drop rows where 'Timestamp' could not be parsed
    sense_data = sense_data.dropna(subset=['Timestamp'])

    # Ensure unique timestamps after parsing
    sense_data = sense_data.drop_duplicates(subset=['Timestamp'])
    
    # Extract the date part for filtering purposes
    sense_data['Date'] = sense_data['Timestamp'].dt.date

    # Convert filtering dates to datetime.date for comparison
    start_unoccupied = pd.to_datetime('2023-12-17').date()
    end_unoccupied = pd.to_datetime('2024-01-02').date()
    start_occupied = pd.to_datetime('2024-01-03').date()
    end_occupied = pd.to_datetime('2024-02-03').date()

    # Filter the data based on the occupancy parameter
    if occupancy == 'unoccupied':
        filtered_data = sense_data[(sense_data['Date'] >= start_unoccupied) & (sense_data['Date'] < end_unoccupied)]
    else:
        filtered_data = sense_data[(sense_data['Date'] >= start_occupied) & (sense_data['Date'] < end_occupied)]

    # Interpolate missing values
    filtered_data = filtered_data.interpolate()

    # Calculate the length of data needed
    len_data = int(duration * 3600 / sampling_rate)
    len_gt = int(pred_hrz * 3600 / sampling_rate)
    
    # Resample the data to the desired sampling rate if needed
    if sampling_rate != 300:
        filtered_data = filtered_data.set_index('Timestamp')
        numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns
        filtered_data = filtered_data[numeric_columns].resample(f'{sampling_rate}S').mean().reset_index()

    # Ensure no duplicates after resampling
    filtered_data = filtered_data.drop_duplicates(subset=['Timestamp'])

    # Ensure timestamps are consecutive
    expected_interval = pd.Timedelta(seconds=sampling_rate)
    actual_intervals = filtered_data['Timestamp'].diff().dropna()
    if not (actual_intervals == expected_interval).all():
        all_timestamps = pd.date_range(start=filtered_data['Timestamp'].min(), end=filtered_data['Timestamp'].max(), freq=f'{sampling_rate}S')
        filtered_data = filtered_data.set_index('Timestamp').reindex(all_timestamps).interpolate().reset_index()
        filtered_data.rename(columns={'index': 'Timestamp'}, inplace=True)

    # Calculate total number of rows
    total_size = len(filtered_data)

    # Calculate start points based on the new algorithm
    max_start_point = total_size - (len_data + len_gt)
    interval = max_start_point // (batch_number - 1)
    start_points = [i * interval for i in range(batch_number)]
    start_points[-1] = max_start_point

    data_start = start_points[batch_id]
    data_end = data_start + len_data
    test_data_start = data_end
    test_data_end = test_data_start + len_gt

    # Ensure we have enough data
    if total_size < len_data + len_gt:
        raise ValueError("Not enough data available for the specified duration and prediction horizon")

    # Convert to numpy arrays
    data = filtered_data[['Active Power', 'Timestamp']].values[data_start:data_end]
    test_data = filtered_data[['Active Power', 'Timestamp']].values[test_data_start:test_data_end]
    # import pdb; pdb.set_trace()
    return data, test_data

import pdb 

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

def sample_by_season(df, dataset='uci'):
    from config import house_ids_uci, house_ids_pecan, house_ids_umass
    if dataset == 'uci':
        house_id_list = house_ids_uci
        start = 4e4
    elif dataset == 'umass':
        house_id_list = house_ids_umass
        start = 0
    else:
        house_id_list = house_ids_pecan
        start = 0
    
    offset = 4*24*10
    period = (len(df)-offset-start) // 16
    whole_indices, whole_seasons = {}, {}
    
    for h_i in range(len(house_id_list)):
        indices = []
        for i in range(0, 16):
            indices.extend(sorted(np.random.randint(i*period+start, (i+1)*period+start, 1).tolist()))
        _df = df.copy()
        _df['Month'] = _df['TS'].dt.month
        
        # Apply the helper function to get the season for the specified indices
        seasons = [get_season(_df.loc[idx, 'Month']) for idx in indices]
        whole_indices[house_id_list[h_i]] = indices
        whole_seasons[house_id_list[h_i]] = seasons
    
    indices_df = pd.DataFrame(whole_indices)
    season_df = pd.DataFrame(whole_seasons)
    indices_df.to_csv(f'./data/{dataset}_indices.csv',index=False)
    season_df.to_csv(f'./data/{dataset}_season.csv', index=False)
    return whole_indices, whole_seasons

def get_uci_electricity_data(duration, pred_hrz, sampling_rate, house_id=1, occupancy=None, batch_id=0, batch_number=10, data_df=None):
    from config import elec_uci_indices, elec_uci_season
    id_num = house_id
    assert house_id >= 1 and house_id <= 370
    house_id = str(house_id)
    house_id = 'MT_'+'0'*(3-len(house_id)) + house_id
    # Read the data, considering the specific delimiter and the first line as header
    # df = pd.read_csv(file_path, delimiter=';', header=0)
    assert data_df is not None
    df = data_df.copy()
    df.rename(columns={df.columns[0]: 'TS'}, inplace=True)

    df = df[['TS', house_id]]

    # Set MT_001 to float
    df[house_id] = df[house_id].str.replace(',', '.').astype(float)

    df['TS'] = pd.to_datetime(df['TS'])
    # print(sample_by_season(df))
    # pdb.set_trace()
    csv_path = './data/uci_indices.csv'
    df_indices = pd.read_csv(csv_path)

    df = df.iloc[df_indices[str(id_num)][batch_id]:-1,:]
    
    df.set_index('TS', inplace=True)

    # resampled_df = df.resample(f'{sampling_rate}s').asfreq().fillna(0).reset_index()
    resampled_df = df.resample(f'{sampling_rate}s').mean().interpolate(method='time').fillna(0).reset_index()
    
    # search for the first non-zero row
    first_non_zero_index = resampled_df[resampled_df[house_id] != 0].index[0]

    # Calculate the length of data needed
    len_data = int(duration * 3600 / sampling_rate)
    len_gt = int(pred_hrz * 3600 / sampling_rate)
    start_points = first_non_zero_index 

    data = resampled_df[[house_id, 'TS']].values[start_points:start_points+len_data]
    test_data = resampled_df[[house_id, 'TS']].values[start_points+len_data:start_points+len_data+len_gt]

    return data, test_data

def get_ecobee_temp_data(duration, pred_hrz, sampling_rate, house_id=1, occupancy=None, batch_id=0, data_df=None):
    # Read the text file into a DataFrame
    file_path = 'data/combined_thermostat_data.csv'  # Assuming the file is saved with this name
    start_points_file = 'data/start_points.csv'

    if data_df is None:
        # Read the data
        df = pd.read_csv(file_path)
    else:
        df = data_df

    unique_ids = df['id'].unique()
    house_id_str = unique_ids[house_id-1]
    
    if not os.path.exists(start_points_file):
        # Initialize the starting points file if it doesn't exist
        start_points = pd.DataFrame(columns=['house_id', 'month', 'start_point'])
    else:
        # Load existing starting points
        start_points = pd.read_csv(start_points_file)

    # Check if starting points exist for the given house_id
    if not (start_points['house_id'] == house_id_str).any():
        # Sample starting points for each month
        sampled_points = []
        for month in range(1, 13):
            month_data = df[(df['id'] == house_id_str) & (pd.to_datetime(df['time']).dt.month == month)]
            if month_data.empty:
                continue
            
            if month == 12:
                # Ensure the starting point is at least a week before the end of the month
                end_of_month = pd.to_datetime(month_data['time']).max()
                valid_end_date = end_of_month - pd.Timedelta(days=16)
                month_data = month_data[pd.to_datetime(month_data['time']) <= valid_end_date]
            
            sampled_point = month_data['time'].sample(n=1).values[0]
            sampled_points.append([house_id_str, month, sampled_point])
        
        # Save the starting points
        new_start_points = pd.DataFrame(sampled_points, columns=['house_id', 'month', 'start_point'])
        start_points = pd.concat([start_points, new_start_points], ignore_index=True)
        start_points.to_csv(start_points_file, index=False)
    else:
        # Retrieve existing starting points
        sampled_points = start_points[start_points['house_id'] == house_id_str]

    # Ensure sampled_points is a DataFrame with proper column names
    sampled_points = pd.DataFrame(sampled_points, columns=['house_id', 'month', 'start_point'])

    # Debug statement to print the sampled_points DataFrame
    #print("Sampled Points DataFrame:\n", sampled_points)

    # Get the starting point for the given batch_id
    sampled_points = sampled_points.sort_values(by=['month'])
    start_point = sampled_points.iloc[batch_id % len(sampled_points)]['start_point']
    
    # Filter data for the house and start from the sampled point
    house_data = df[(df['id'] == house_id_str) & (df['time'] >= start_point)].copy()
    
    # Convert 'time' column to datetime
    house_data['time'] = pd.to_datetime(house_data['time'])
    
    # Check and resample if necessary
    if sampling_rate != 300:
        house_data = house_data.set_index('time').resample(f'{sampling_rate}s').agg({
            'temperature': 'mean',
            'state': 'first',
            'id': 'first'
        }).reset_index()

    #interpolate the missing values
    house_data['temperature'] = house_data['temperature'].interpolate()

    # Calculate the length of data needed
    len_data = int(duration * 3600 / sampling_rate)
    len_gt = int(pred_hrz * 3600 / sampling_rate)

    # Extract the required data and test data
    data = house_data[['temperature', 'time']].iloc[:len_data].values
    #print(f"data head for batch {batch_id}", data[0])
    test_data = house_data[['temperature', 'time']].iloc[len_data:len_data+len_gt].values

    return data, test_data

def get_pecan_data(duration, pred_hrz, sampling_rate, house_id=1, occupancy=None, batch_id=0, batch_number=10, data_df=None):

    id_num = house_id
    house_id = str(house_id)
    house_id = 'MT_'+'0'*(3-len(house_id)) + house_id
    # Read the data, considering the specific delimiter and the first line as header
    # df = pd.read_csv(file_path, delimiter=';', header=0)
    assert data_df is not None
    df = data_df.copy()
    df.rename(columns={df.columns[0]: 'TS'}, inplace=True)

    df['TS'] = pd.to_datetime(df['TS'])
    
    # print(sample_by_season(df, dataset='pecan'))
    df = df[['TS', house_id]]
    
    csv_path = './data/pecan_indices.csv'
    df_indices = pd.read_csv(csv_path)

    df = df.iloc[df_indices[str(id_num)][batch_id]:-1,:]
    
    df.set_index('TS', inplace=True)

    # resampled_df = df.resample(f'{sampling_rate}s').asfreq().fillna(0).reset_index()
    resampled_df = df.resample(f'{sampling_rate}s').mean().interpolate(method='time').fillna(0).reset_index()
    
    # search for the first non-zero row
    first_non_zero_index = resampled_df[resampled_df[house_id] != 0].index[0]

    # Calculate the length of data needed
    len_data = int(duration * 3600 / sampling_rate)
    len_gt = int(pred_hrz * 3600 / sampling_rate)
    start_points = first_non_zero_index 

    data = resampled_df[[house_id, 'TS']].values[start_points:start_points+len_data]
    test_data = resampled_df[[house_id, 'TS']].values[start_points+len_data:start_points+len_data+len_gt]

    return data, test_data

def get_umass_data(duration, pred_hrz, sampling_rate, house_id=1, occupancy=None, batch_id=0, batch_number=10, data_df=None):

    id_num = house_id
    house_id = str(house_id)
    house_id = 'MT_'+'0'*(3-len(house_id)) + house_id
    # Read the data, considering the specific delimiter and the first line as header
    # df = pd.read_csv(file_path, delimiter=';', header=0)
    assert data_df is not None
    df = data_df.copy()
    df.rename(columns={df.columns[0]: 'TS'}, inplace=True)

    df['TS'] = pd.to_datetime(df['TS'])
    
    # print(sample_by_season(df, dataset='umass'))
    # pdb.set_trace()
    df = df[['TS', house_id]]
    
    csv_path = './data/umass_indices.csv'
    df_indices = pd.read_csv(csv_path)

    df = df.iloc[df_indices[str(id_num)][batch_id]:-1,:]
    
    df.set_index('TS', inplace=True)

    # resampled_df = df.resample(f'{sampling_rate}s').asfreq().fillna(0).reset_index()
    resampled_df = df.resample(f'{sampling_rate}s').mean().interpolate(method='time').fillna(0).reset_index()
    
    # search for the first non-zero row
    if len(resampled_df[resampled_df[house_id] != 0]) == 0:
        first_non_zero_index = 0
    else:
        first_non_zero_index = resampled_df[resampled_df[house_id] != 0].index[0]

    # Calculate the length of data needed
    len_data = int(duration * 3600 / sampling_rate)
    len_gt = int(pred_hrz * 3600 / sampling_rate)
    start_points = first_non_zero_index 

    data = resampled_df[[house_id, 'TS']].values[start_points:start_points+len_data]
    test_data = resampled_df[[house_id, 'TS']].values[start_points+len_data:start_points+len_data+len_gt]
    # pdb.set_trace()
    return data, test_data



def get_elecdemand_data(duration, pred_hrz, sampling_rate=None, house_id=1, batch_id=0, batch_number=10, data_df=None):
    """
    Extracts electricity demand data from the 'Salesforce/lotsa_data' dataset, 'elecdemand' subset.

    Parameters:
    - duration: The duration of the data in hours.
    - pred_hrz: The prediction horizon in hours.
    - sampling_rate: Sampling rate in seconds. If None, the original sampling rate from the dataset is used.
    - house_id: The ID of the house (defaults to 1).
    - batch_id: Batch ID to extract from data (defaults to 0).
    - batch_number: Number of batches (defaults to 10).

    Returns:
    - data: The extracted data for the specified duration.
    - test_data: The test data for the prediction horizon.
    """

    # Access the 'train' split of the dataset
    train_data = data_df['train']

    # Extract the start timestamp, frequency, and values for the selected house_id
    start_timestamp = pd.to_datetime(train_data['start'][house_id-1])  # Convert start to datetime
    freq = train_data['freq'][house_id-1]  # Get frequency (e.g., '30T')
    values = np.array(train_data['target'][house_id-1])  # Values array

    # Generate the timestamps using pandas date_range with the given start and frequency
    timestamps = pd.date_range(start=start_timestamp, periods=len(values), freq=freq)

    # Create a DataFrame with values and timestamps
    df = pd.DataFrame({'values': values, 'timestamps': timestamps})

    # Handle resampling if a sampling rate is provided
    if sampling_rate:
        # Resample the data to the specified sampling rate
        df.set_index('timestamps', inplace=True)
        df = df.resample(f'{sampling_rate}S').mean().interpolate(method='time').fillna(0).reset_index()

    # Get the first non-zero value as the start point
    first_non_zero_index = df[df['values'] != 0].index[0]
    
    # Calculate the number of rows to extract based on duration and prediction horizon
    len_data = int(duration * 3600 / (sampling_rate or pd.to_timedelta(freq).seconds))
    len_gt = int(pred_hrz * 3600 / (sampling_rate or pd.to_timedelta(freq).seconds))

    # Calculate batch size in terms of number of samples
    total_size = len(df)
    max_start_point = total_size - (len_data + len_gt)
    interval = max_start_point // (batch_number - 1)

    # Create start points
    start_points = [i * interval for i in range(batch_number)]
    start_points[-1] = max_start_point

    # Select start and end points for data and test data based on batch_id
    data_start = start_points[batch_id]
    data_end = data_start + len_data
    test_data_start = data_end
    test_data_end = test_data_start + len_gt

    # Extract data and test data
    data = df[['values', 'timestamps']].values[data_start:data_end]
    test_data = df[['values', 'timestamps']].values[test_data_start:test_data_end]

    return data, test_data


def generate_datetime_list(start_datetime, increase, num_steps, offset=0):
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta 
    # Ensure start_datetime is a pandas Timestamp and has time components
    if not isinstance(start_datetime, pd.Timestamp):
        raise ValueError("start_datetime must be a pandas Timestamp.")
    
    # Set time to 00:00:00 if start_datetime does not include hours, minutes, and seconds
    if start_datetime.hour == 0 and start_datetime.minute == 0 and start_datetime.second == 0:
        start_datetime = start_datetime.replace(hour=0, minute=0, second=0)
    
    # Extract the number and unit from 'increase'
    if increase[:-1].isdigit():
        n = int(increase[:-1])
        unit = increase[-1]
    else:
        n = 1  # Default increment if no number is provided
        unit = increase

    # Determine the increment based on the unit
    if unit == 'H':
        increment = timedelta(hours=n)
    elif unit == 'T':
        increment = timedelta(minutes=n)
    elif unit == 'D':
        increment = timedelta(days=n)
    elif unit == 'M':
        increment = relativedelta(months=n)
    elif unit == 'A-DEC':
        increment = relativedelta(years=n)
    elif unit == 'W-SUN':
        increment = relativedelta(weeks=n)
    else:
        raise ValueError("Invalid increase value. Must be in ['H', 'T', 'D', 'M', 'A-DEC', 'W-SUN'] with optional 'n' prefix.")

    # Generate the list of datetime values
    datetime_list = []
    for i in range(num_steps):
        datetime_list.append(start_datetime + (offset + i) * increment)
    
    return datetime_list

def get_subseasonal_data(duration, pred_hrz, sampling_rate=None, house_id=1, batch_id=0, batch_number=10, data_df=None):
    """
    Extracts weather data from the 'Salesforce/lotsa_data' dataset, 'subseasonal' subset.

    Parameters:
    - duration: The duration of the data in hours.
    - pred_hrz: The prediction horizon in hours.
    - sampling_rate: Sampling rate in seconds. If None, the original sampling rate from the dataset is used.
    - house_id: The ID of the house (defaults to 1).
    - batch_id: Batch ID to extract from data (defaults to 0).
    - batch_number: Number of batches (defaults to 10).

    Returns:
    - data: The extracted data for the specified duration.
    - test_data: The test data for the prediction horizon.
    """
    assert batch_id <= 861
    dataset = data_df['train']
    dataset_pd = dataset.to_pandas()
    start_datetime = dataset_pd['start'][batch_id]
    increase =  dataset_pd['freq'][0]
    len_data = int(duration * 3600 / (sampling_rate))
    len_gt = int(pred_hrz * 3600 / (sampling_rate))
    tot_len = len_gt+len_data
    datetime_list = generate_datetime_list(start_datetime, increase, num_steps=tot_len)
    data = dataset_pd['target'][batch_id][0][:tot_len]
    data_all =  np.stack([data, datetime_list]).T
    return data_all[:len_data,:], data_all[len_data: tot_len, :]

def get_pems04_data(duration, pred_hrz, sampling_rate=None, house_id=1, batch_id=0, batch_number=10, data_df=None):
    """
    Extracts weather data from the 'Salesforce/lotsa_data' dataset, 'subseasonal' subset.

    Parameters:
    - duration: The duration of the data in hours.
    - pred_hrz: The prediction horizon in hours.
    - sampling_rate: Sampling rate in seconds. If None, the original sampling rate from the dataset is used.
    - house_id: The ID of the house (defaults to 1).
    - batch_id: Batch ID to extract from data (defaults to 0).
    - batch_number: Number of batches (defaults to 10).

    Returns:
    - data: The extracted data for the specified duration.
    - test_data: The test data for the prediction horizon.
    """
    assert batch_id <= 307
    dataset = data_df['train']
    dataset_pd = dataset.to_pandas()
    start_datetime = dataset_pd['start'][batch_id]
    increase =  dataset_pd['freq'][0]
    len_data = int(duration * 3600 / (sampling_rate))
    len_gt = int(pred_hrz * 3600 / (sampling_rate))
    tot_len = len_gt+len_data
    datetime_list = generate_datetime_list(start_datetime, increase, num_steps=tot_len)
    data = dataset_pd['target'][batch_id][0][:tot_len]
    data_all =  np.stack([data, datetime_list]).T
    return data_all[:len_data,:], data_all[len_data: tot_len, :]

def get_loop_seattle_data(duration, pred_hrz, sampling_rate=None, house_id=1, batch_id=0, batch_number=10, data_df=None):
    """
    Extracts weather data from the 'Salesforce/lotsa_data' dataset, 'subseasonal' subset.

    Parameters:
    - duration: The duration of the data in hours.
    - pred_hrz: The prediction horizon in hours.
    - sampling_rate: Sampling rate in seconds. If None, the original sampling rate from the dataset is used.
    - house_id: The ID of the house (defaults to 1).
    - batch_id: Batch ID to extract from data (defaults to 0).
    - batch_number: Number of batches (defaults to 10).

    Returns:
    - data: The extracted data for the specified duration.
    - test_data: The test data for the prediction horizon.
    """
    assert batch_id <= 323
    dataset = data_df['train']
    dataset_pd = dataset.to_pandas()
    start_datetime = dataset_pd['start'][batch_id]
    increase =  dataset_pd['freq'][0]
    len_data = int(duration * 3600 / (sampling_rate))
    len_gt = int(pred_hrz * 3600 / (sampling_rate))
    tot_len = len_gt+len_data
    datetime_list = generate_datetime_list(start_datetime, increase, num_steps=tot_len)
    data = dataset_pd['target'][batch_id][:tot_len]
    data_all =  np.stack([data, datetime_list]).T
    return data_all[:len_data,:], data_all[len_data: tot_len, :]

def get_rlp_data(duration, pred_hrz, sampling_rate=None, house_id=1, batch_id=0, batch_number=10, data_df=None):
    """
    Extracts weather data from the 'Salesforce/lotsa_data' dataset, 'subseasonal' subset.

    Parameters:
    - duration: The duration of the data in hours.
    - pred_hrz: The prediction horizon in hours.
    - sampling_rate: Sampling rate in seconds. If None, the original sampling rate from the dataset is used.
    - house_id: The ID of the house (defaults to 1).
    - batch_id: Batch ID to extract from data (defaults to 0).
    - batch_number: Number of batches (defaults to 10).

    Returns:
    - data: The extracted data for the specified duration.
    - test_data: The test data for the prediction horizon.
    """
    assert batch_id <= 271
    dataset = data_df['train']
    dataset_pd = dataset.to_pandas()
    start_datetime = dataset_pd['start'][batch_id]
    increase =  dataset_pd['freq'][0]
    len_data = int(duration * 3600 / (sampling_rate))
    len_gt = int(pred_hrz * 3600 / (sampling_rate))
    tot_len = len_gt+len_data
    datetime_list = generate_datetime_list(start_datetime, increase, num_steps=tot_len)
    data = dataset_pd['target'][batch_id][0][:tot_len]
    data_all =  np.stack([data, datetime_list]).T
    return data_all[:len_data,:], data_all[len_data: tot_len, :]

def get_covid_data(duration, pred_hrz, sampling_rate=None, house_id=1, batch_id=0, batch_number=10, data_df=None):
    assert batch_id <= 271
    dataset = data_df['train']
    dataset_pd = dataset.to_pandas()
    start_datetime = dataset_pd['start'][batch_id]
    increase =  dataset_pd['freq'][0]
    len_data = int(duration * 3600 / (sampling_rate))
    len_gt = int(pred_hrz * 3600 / (sampling_rate))
    tot_len = len_gt+len_data
    datetime_list = generate_datetime_list(start_datetime, increase, num_steps=tot_len)
    data = dataset_pd['target'][batch_id][:tot_len]
    data_all =  np.stack([data, datetime_list]).T
    return data_all[:len_data,:], data_all[len_data: tot_len, :]

def get_c2000_data(duration, pred_hrz, sampling_rate=None, house_id=1, batch_id=0, batch_number=10, data_df=None):
    assert batch_id <= 300
    dataset_pd = data_df
    start_datetime = dataset_pd['start'][batch_id]
    increase =  dataset_pd['freq'][0]
    len_data = int(duration * 3600 / (sampling_rate))
    len_gt = int(pred_hrz * 3600 / (sampling_rate))
    tot_len = len_gt+len_data
    datetime_list = generate_datetime_list(start_datetime, increase, num_steps=tot_len)
    if isinstance(dataset_pd['target'][batch_id][0], np.ndarray):
        data = dataset_pd['target'][batch_id][0][:tot_len]
    else:
        data = dataset_pd['target'][batch_id][:tot_len]
    data_all =  np.stack([data, datetime_list]).T
    return data_all[:len_data,:], data_all[len_data: tot_len, :]

def get_restaurant_data(duration, pred_hrz, sampling_rate=None, house_id=1, batch_id=0, batch_number=10, data_df=None):
    assert batch_id <= 200
    dataset_pd = data_df
    start_datetime = dataset_pd['start'][batch_id]
    increase =  dataset_pd['freq'][0]
    len_data = int(duration * 3600 / (sampling_rate))
    len_gt = int(pred_hrz * 3600 / (sampling_rate))
    tot_len = len_gt+len_data
    datetime_list = generate_datetime_list(start_datetime, increase, num_steps=tot_len)
    if isinstance(dataset_pd['target'][batch_id][0], np.ndarray):
        data = dataset_pd['target'][batch_id][0][:tot_len]
    else:
        data = dataset_pd['target'][batch_id][:tot_len]
    
    if len(data) < tot_len:
        data = np.append(data, [0] * (tot_len - len(data)))

    not_nan = ~np.isnan(data)
    x = np.arange(len(data))
    # Applying linear interpolation
    linear_interpolator = interp1d(x[not_nan], data[not_nan], kind='linear', fill_value="extrapolate")
    data = linear_interpolator(x)
    
    data_all =  np.stack([data, datetime_list]).T
    return data_all[:len_data,:], data_all[len_data: tot_len, :]

def get_air_data(duration, pred_hrz, sampling_rate=None, house_id=1, batch_id=0, batch_number=10, data_df=None):
    assert batch_id <= 200
    dataset_pd = data_df
    start_datetime = dataset_pd['start'][batch_id]
    increase =  dataset_pd['freq'][0]
    len_data = int(duration * 3600 / (sampling_rate))
    len_gt = int(pred_hrz * 3600 / (sampling_rate))
    tot_len = len_gt+len_data
    datetime_list = generate_datetime_list(start_datetime, increase, num_steps=tot_len)
    if isinstance(dataset_pd['target'][batch_id][0], np.ndarray):
        data = dataset_pd['target'][batch_id][0][:tot_len]
    else:
        data = dataset_pd['target'][batch_id][:tot_len]
    
    if len(data) < tot_len:
        data = np.append(data, [0] * (tot_len - len(data)))

    not_nan = ~np.isnan(data)
    x = np.arange(len(data))
    # Applying linear interpolation
    linear_interpolator = interp1d(x[not_nan], data[not_nan], kind='linear', fill_value="extrapolate")
    data = linear_interpolator(x)
    
    data_all =  np.stack([data, datetime_list]).T
    return data_all[:len_data,:], data_all[len_data: tot_len, :]

if __name__ == '__main__':
    file_path = './LD2011_2014.txt'
    df = pd.read_csv(file_path, delimiter=';', header=0)
    house_id = 'MT_001'
    df = df[['TS', house_id]]

    # Set MT_001 to float
    df[house_id] = df[house_id].str.replace(',', '.').astype(float)

    df['TS'] = pd.to_datetime(df['TS'])
    df = df.iloc[:-1,:]
    sample_by_season(df)
    get_uci_electricity_data(duration=24, pred_hrz=4, sampling_rate=900, data_df=data_df)




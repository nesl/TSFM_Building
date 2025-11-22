"""
Thermostat Data Processing Script

This script processes monthly thermostat temperature data from NetCDF files and creates
two datasets:

1. combined_thermostat_data.csv (Test Set):
   - Contains data from the top 8 houses per state with the least NaN values
   - Used for testing both ARIMA models and Time Series Foundation Models (TSFMs)

2. training_thermostat_data.csv (Training Set):
   - Contains data from the next 32 houses per state (ranked 9-40 by NaN counts)
   - Used for pretraining ARIMA models

The script also verifies that there is no data leakage between the training and test sets.
"""

import xarray as xr
import pandas as pd
import numpy as np


# List of file names for each month
file_names = ['Jan_clean.nc', 'Feb_clean.nc', 'Mar_clean.nc', 'Apr_clean.nc', 'May_clean.nc', 
              'Jun_clean.nc', 'Jul_clean.nc', 'Aug_clean.nc', 'Sep_clean.nc', 'Oct_clean.nc', 
              'Nov_clean.nc', 'Dec_clean.nc']


def load_files(file_names):
    """
    Load NetCDF files and accumulate NaN counts for each house across all months.
    
    Args:
        file_names (list): List of NetCDF file names to load
        
    Returns:
        tuple: (data_sets, house_nan_counts, state_info)
    """
    data_sets = []
    house_nan_counts = {}
    state_info = {}
    
    # Iterate over each file and accumulate NaN counts
    for file_index, file_name in enumerate(file_names):
        print(f"Loading file {file_index + 1}/{len(file_names)}: {file_name}")
        ds = xr.open_dataset(file_name)
        ds = ds[['Thermostat_Temperature', 'time', 'State', 'id']]  # Keep only necessary variables
        data_sets.append(ds)
        
        for house_id in ds.id.values:
            house_data = ds.sel(id=house_id)
            states = house_data.State.values
            unique_states = np.unique(states)
            valid_states = [state for state in unique_states if isinstance(state, str) and state]
            
            if valid_states:
                state_info[house_id] = valid_states[0]
            
            if house_id not in house_nan_counts:
                house_nan_counts[house_id] = 0
            house_nan_counts[house_id] += house_data.Thermostat_Temperature.isnull().sum().item()
    
    print("Files loaded and NaN counts aggregated.")
    return data_sets, house_nan_counts, state_info


def get_top_houses(house_nan_counts, state_info, n=8):
    """
    Select top houses with the least NaN counts for each state.
    
    Args:
        house_nan_counts (dict): Dictionary mapping house IDs to their NaN counts
        state_info (dict): Dictionary mapping house IDs to their states
        n (int): Number of houses to select per state
        
    Returns:
        list: List of selected house IDs
    """
    selected_houses = []
    states = np.unique(list(state_info.values()))
    
    for state in states:
        state_houses = [house for house in house_nan_counts if state_info[house] == state]
        sorted_houses = sorted(state_houses, key=lambda x: house_nan_counts[x])
        selected_houses.extend(sorted_houses[:n])
    
    print("Top houses selected based on least NaN values for the entire year.")
    return selected_houses


def get_training_houses(house_nan_counts, state_info, test_set_size=8, train_set_size=32):
    """
    Select the next batch of houses for the training set from each state.
    This function sorts houses by their NaN counts and skips the initial houses
    that were selected for the test set.
    
    Args:
        house_nan_counts (dict): A dictionary mapping house IDs to their NaN counts
        state_info (dict): A dictionary mapping house IDs to their respective states
        test_set_size (int): The number of houses selected for the test set per state
        train_set_size (int): The number of houses to select for the training set per state
        
    Returns:
        list: A list of house IDs selected for the training set
    """
    training_houses = []
    states = np.unique(list(state_info.values()))
    
    # Define the start and end indices for slicing the sorted house list
    start_index = test_set_size
    end_index = test_set_size + train_set_size
    
    for state in states:
        # Filter houses belonging to the current state
        state_houses = [house for house in house_nan_counts if state_info[house] == state]
        
        # Sort the houses based on their NaN count (ascending)
        sorted_houses = sorted(state_houses, key=lambda x: house_nan_counts[x])
        
        # Select the slice of houses for the training set
        training_houses.extend(sorted_houses[start_index:end_index])
    
    print(f"Selected {train_set_size} training houses from each state, starting after the first {test_set_size}.")
    return training_houses


def check_for_data_leakage(train_file, test_file):
    """
    Check for data leakage between training and testing datasets by comparing house IDs.
    
    Args:
        train_file (str): The file path for the training data CSV
        test_file (str): The file path for the testing data CSV
    """
    print(f"Checking for data leakage between '{train_file}' and '{test_file}'...")
    
    try:
        # Read the training and testing datasets
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        print("Successfully loaded both CSV files.")
        
        # Get the set of unique house IDs from each DataFrame
        train_ids = set(train_df['id'].unique())
        test_ids = set(test_df['id'].unique())
        
        print(f"Found {len(train_ids)} unique house IDs in the training set.")
        print(f"Found {len(test_ids)} unique house IDs in the test set.")
        
        # Find the intersection of the two sets of IDs
        overlapping_ids = train_ids.intersection(test_ids)
        
        # Check if the intersection is empty and report the result
        if not overlapping_ids:
            print("\n--- Verification Result ---")
            print("✅ No data leakage detected. The house IDs in the training and test sets are mutually exclusive.")
            print("---------------------------\n")
        else:
            print("\n--- Verification Result ---")
            print(f"🚨 Warning: Data leakage detected!")
            print(f"The following {len(overlapping_ids)} house ID(s) are present in BOTH the training and test sets:")
            for house_id in sorted(list(overlapping_ids)):
                print(f"  - {house_id}")
            print("---------------------------\n")
    except FileNotFoundError as e:
        print(f"\nError: Could not find one of the files.")
        print(f"Details: {e}")
        print("Please make sure both CSV files are in the same directory as this script.")
    except KeyError as e:
        print(f"\nError: A required column is missing from one of the CSV files.")
        print(f"Details: Missing column {e}. Please ensure both files have an 'id' column.")


def main():
    """Main execution function."""
    
    # Load the files and get the NaN counts and state information
    print("=" * 60)
    print("STEP 1: Loading data and computing NaN counts")
    print("=" * 60)
    data_sets, house_nan_counts, state_info = load_files(file_names)
    
    # Create test set
    print("\n" + "=" * 60)
    print("STEP 2: Creating test set (combined_thermostat_data.csv)")
    print("=" * 60)
    selected_houses = get_top_houses(house_nan_counts, state_info)
    
    # Initialize a dictionary to store the data for the selected houses
    house_data = {house_id: [] for house_id in selected_houses}
    
    # Process the data for the selected houses using the loaded datasets
    for file_index, ds in enumerate(data_sets):
        print(f"Processing data from file {file_index + 1}/{len(file_names)} for selected houses")
        
        # Extract data for the selected houses
        for house_id in selected_houses:
            house_series = ds.sel(id=house_id).Thermostat_Temperature.to_pandas()
            house_data[house_id].append(house_series)
    
    # Combine the monthly data into a single DataFrame for each house and save to CSV
    print("Combining data for each house into a yearly dataset.")
    final_data = pd.DataFrame()
    
    for house_id, data_list in house_data.items():
        yearly_data = pd.concat(data_list, axis=0)
        yearly_data = yearly_data.reset_index()
        yearly_data['id'] = house_id
        yearly_data['temperature'] = yearly_data[0]
        yearly_data['state'] = state_info[house_id]
        yearly_data = yearly_data.drop(columns=[0])
        final_data = pd.concat([final_data, yearly_data], axis=0)
    
    # Save the final combined DataFrame to a CSV file
    final_data.to_csv('combined_thermostat_data.csv', index=False)
    print("Data saved to 'combined_thermostat_data.csv'.")
    
    # Create training set
    print("\n" + "=" * 60)
    print("STEP 3: Creating training set (training_thermostat_data.csv)")
    print("=" * 60)
    training_houses = get_training_houses(house_nan_counts, state_info, test_set_size=8, train_set_size=32)
    
    # Initialize a dictionary to store the data for the selected training houses
    training_house_data = {house_id: [] for house_id in training_houses}
    
    # Process the data for the selected training houses using the loaded datasets
    for file_index, ds in enumerate(data_sets):
        print(f"Processing data from file {file_index + 1}/{len(file_names)} for training houses")
        
        # Extract data for the selected training houses
        for house_id in training_houses:
            # Ensure the house_id exists in the current dataset slice
            if house_id in ds.id.values:
                house_series = ds.sel(id=house_id).Thermostat_Temperature.to_pandas()
                training_house_data[house_id].append(house_series)
    
    # Combine the monthly data into a single DataFrame for each training house
    print("Combining data for each training house into a yearly dataset.")
    final_training_data = pd.DataFrame()
    
    for house_id, data_list in training_house_data.items():
        if not data_list:
            print(f"Warning: No data found for training house {house_id}. Skipping.")
            continue
        
        # Concatenate all monthly Series for the current house
        yearly_data = pd.concat(data_list, axis=0)
        yearly_data = yearly_data.reset_index()
        
        # Rename columns and add metadata
        yearly_data.columns = ['datetime', 'temperature']  # Assuming the index is datetime
        yearly_data['id'] = house_id
        yearly_data['state'] = state_info[house_id]
        
        # Append to the final training DataFrame
        final_training_data = pd.concat([final_training_data, yearly_data], axis=0, ignore_index=True)
    
    # Verification Prints
    print("\n--- Data Verification ---")
    # Print the total number of unique house IDs in the final training data
    total_unique_houses = final_training_data['id'].nunique()
    print(f"Total number of unique houses in the training set: {total_unique_houses}")
    
    # Print the number of unique houses per state
    houses_per_state = final_training_data.groupby('state')['id'].nunique()
    print("\nNumber of unique houses per state:")
    print(houses_per_state)
    print("-------------------------\n")
    
    # Save the final combined training DataFrame to a new CSV file
    output_filename = 'training_thermostat_data.csv'
    final_training_data.to_csv(output_filename, index=False)
    print(f"Training data successfully saved to '{output_filename}'.")
    
    # Check for data leakage
    print("\n" + "=" * 60)
    print("STEP 4: Verifying no data leakage between sets")
    print("=" * 60)
    check_for_data_leakage('training_thermostat_data.csv', 'combined_thermostat_data.csv')


if __name__ == "__main__":
    main()
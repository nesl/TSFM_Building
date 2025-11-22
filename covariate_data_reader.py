import pandas as pd
import numpy as np
import os
import pdb 

def read_csvs_to_dfs(main_output_directory):
    all_houses_dict = {}
    
    # Iterate over each subdirectory within the main directory
    for subdirectory in os.listdir(main_output_directory):
        sub_output_directory = os.path.join(main_output_directory, subdirectory)
        
        # Skip if it's not a directory
        if not os.path.isdir(sub_output_directory):
            continue
        
        # Extract house group from the subdirectory name and convert to integer
        house_group = int(subdirectory.split("_")[-1])
        
        # Initialize the dictionary for this house group if it doesn't exist
        if house_group not in all_houses_dict:
            all_houses_dict[house_group] = {}
        
        # Iterate over each CSV file within the subdirectory
        for filename in os.listdir(sub_output_directory):
            if filename.endswith(".csv"):
                # Construct the full file path
                file_path = os.path.join(sub_output_directory, filename)
                
                # Extract house_id from the filename
                house_id = filename.split("_")[-1].replace(".csv", "")
                
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                
                # Store the DataFrame in the dictionary under the correct house group
                all_houses_dict[house_group][house_id] = df
                
    return all_houses_dict

def read_csvs_to_dfs_and_add_noise(main_output_directory, epsilon):
    """
    Reads CSVs from a nested directory structure and adds cumulative noise to the data.
    
    Parameters:
        main_output_directory (str): Path to the main directory containing subdirectories of CSVs.
        epsilon (float): Standard deviation of the noise process to be added.

    Returns:
        dict: A nested dictionary containing house groups and their corresponding DataFrames with noise added.
    """
    new_dir = main_output_directory + '_' + str(epsilon) + '/'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir, exist_ok=True)

    all_houses_dict = {}

    # Iterate over each subdirectory within the main directory
    for subdirectory in os.listdir(main_output_directory):
        sub_output_directory = os.path.join(main_output_directory, subdirectory)
        
        # Skip if it's not a directory
        if not os.path.isdir(sub_output_directory):
            continue
        
        # Extract house group from the subdirectory name and convert to integer
        house_group = int(subdirectory.split("_")[-1])
        
        # Initialize the dictionary for this house group if it doesn't exist
        if house_group not in all_houses_dict:
            all_houses_dict[house_group] = {}
        
        # Iterate over each CSV file within the subdirectory
        for filename in os.listdir(sub_output_directory):

            if not os.path.exists(new_dir + subdirectory):
                os.makedirs(new_dir + subdirectory, exist_ok=True)

            if filename.endswith(".csv"):
                # Construct the full file path
                file_path = os.path.join(sub_output_directory, filename)
                
                # Extract house_id from the filename
                house_id = filename.split("_")[-1].replace(".csv", "")
                
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                
                # Add cumulative noise to all numeric columns in the DataFrame
                for col in ['Outdoor_Temperature', 'GHI']:
                    seg_idx = int(len(df[col])*(0.125))
                    noise = np.random.normal(0, epsilon, seg_idx)
                    cumulative_noise = np.cumsum(noise)  # Generate cumulative noise
                    df[col] = np.float64(df[col])
                    # df[col][-seg_idx:] += cumulative_noise  # Add cumulative noise to the column
                    df.loc[len(df)-seg_idx:len(df),col] += cumulative_noise
                
                output_path = new_dir + subdirectory + '/' + filename
                # pdb.set_trace()
                df.to_csv(output_path, index=False)
                

if __name__ == '__main__':
    
    # for e in [5]:
    for e in [0.1, 0.2, 0.5, 1, 2, 5]:
        read_csvs_to_dfs_and_add_noise('./house_data_csvs', epsilon=e)

    # total = 0
    # for k in data:
    #     for k_1 in data[k]:
    #         total += 1
    # print(total)
    # pdb.set_trace()
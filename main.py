import argparse
import pandas as pd  # requires: pip install pandas
import pdb 
from utils import plot_pred, signal_simulate, test_foundation_model, \
    load_foundation_model, save_results_for_model
from config import building_condition,\
                building_dura_pred_sr_tuple, \
                electricity_condition, \
                electricity_dura_pred_sr_tuple, \
                electricity_uci_condition, \
                electricity_uci_dura_pred_sr_tuple, \
                ecobee_condition, \
                ecobee_dura_pred_sr_tuple, \
                pecan_condition, \
                pecan_dura_pred_sr_tuple, \
                umass_condition, \
                umass_dura_pred_sr_tuple, \
                elecdemand_condition, \
                elecdemand_dura_pred_sr_tuple, \
                subseasonal_condition, subseasonal_dura_pred_sr_tuple, \
                pems04_condition, pems04_dura_pred_sr_tuple, \
                loop_seattle_condition, loop_seattle_dura_pred_sr_tuple, \
                rlp_condition, rlp_dura_pred_sr_tuple, \
                covid_condition, covid_dura_pred_sr_tuple, \
                c2000_condition, c2000_dura_pred_sr_tuple, \
                restaurant_condition, restaurant_dura_pred_sr_tuple, \
                air_condition, air_dura_pred_sr_tuple
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from datasets import load_dataset

def main(args):
    if args.real_data == 'building':
        external_condition = building_condition
        data_condition = building_dura_pred_sr_tuple
        data_df = None 
        batch_number = 10
    elif args.real_data == 'electricity':
        external_condition = electricity_condition
        data_condition = electricity_dura_pred_sr_tuple
        data_df = None 
        batch_number = 10
    elif args.real_data == 'electricity_uci':
        external_condition = electricity_uci_condition
        data_condition = electricity_uci_dura_pred_sr_tuple
        # read the data once instead of everytime it is used, which is time consuming
        file_path = './LD2011_2014.txt'
        data_df = pd.read_csv(file_path, delimiter=';', header=0)
        batch_number = 16
    elif args.real_data == 'pecan':
        external_condition = pecan_condition
        data_condition = pecan_dura_pred_sr_tuple
        # read the data once instead of everytime it is used, which is time consuming
        file_path = './data/newyork.csv'
        data_df = pd.read_csv(file_path)
        batch_number = 16
    elif args.real_data == 'umass':
        external_condition = umass_condition
        data_condition = umass_dura_pred_sr_tuple
        # read the data once instead of everytime it is used, which is time consuming
        file_path = './data/merged_2016.csv'
        data_df = pd.read_csv(file_path)
        batch_number = 16
    elif args.real_data == 'ecobee':
        external_condition = ecobee_condition
        data_condition = ecobee_dura_pred_sr_tuple
        file_path = './data/combined_thermostat_data.csv'
        data_df = pd.read_csv(file_path)
        batch_number = 12
    elif args.real_data == 'elecdemand':
        external_condition = elecdemand_condition
        data_condition = elecdemand_dura_pred_sr_tuple
        data_df =  load_dataset("Salesforce/lotsa_data", "elecdemand")
        batch_number = 10
    elif args.real_data == 'subseasonal':
        external_condition = subseasonal_condition
        data_condition = subseasonal_dura_pred_sr_tuple
        data_df =  load_dataset("Salesforce/lotsa_data", "subseasonal")
        batch_number = 200
    elif args.real_data == 'pems04':
        external_condition = pems04_condition
        data_condition = pems04_dura_pred_sr_tuple
        data_df =  load_dataset("Salesforce/lotsa_data", "PEMS04")
        batch_number = 200
    elif args.real_data == 'loop_seattle':
        external_condition = loop_seattle_condition
        data_condition = loop_seattle_dura_pred_sr_tuple
        data_df =  load_dataset("Salesforce/lotsa_data", "LOOP_SEATTLE")
        batch_number = 200
    elif args.real_data == 'rlp':
        external_condition = rlp_condition
        data_condition = rlp_dura_pred_sr_tuple
        data_df =  load_dataset("Salesforce/lotsa_data", "residential_load_power")
        batch_number = 200
    elif args.real_data == 'covid':
        external_condition = covid_condition
        data_condition = covid_dura_pred_sr_tuple
        data_df =  load_dataset("Salesforce/lotsa_data", "covid_deaths")
        batch_number = 200
    elif args.real_data == 'c2000':
        external_condition = c2000_condition
        data_condition = c2000_dura_pred_sr_tuple
        data_df =  load_dataset("Salesforce/lotsa_data", "cmip6_2000")
        dataset = data_df['train']
        data_df = dataset.to_pandas()
        batch_number = 200
    elif args.real_data == 'restaurant':
        external_condition = restaurant_condition
        data_condition = restaurant_dura_pred_sr_tuple
        data_df =  load_dataset("Salesforce/lotsa_data", "restaurant")
        dataset = data_df['train']
        data_df = dataset.to_pandas()
        batch_number = 200
    elif args.real_data == 'air':
        external_condition = air_condition
        data_condition = air_dura_pred_sr_tuple
        data_df =  load_dataset("Salesforce/lotsa_data", "china_air_quality")
        dataset = data_df['train']
        data_df = dataset.to_pandas()
        batch_number = 200
    else:
        raise NotImplementedError("Only building | electricity | electricity_uci | ecobee | elecdemand | subseasonal data are supported!")

    for (duration, pred_hrz, sampling_rate) in tqdm(data_condition):

        args.pre_hrz = pred_hrz
        args.sampling_rate = sampling_rate
        args.duration = duration

        model = load_foundation_model(args, pred_hrz)
        
        for (hvac, occupancy) in external_condition:

            for batch_id in range(batch_number):
                print(">>>>>>> ", args.sampling_rate, hvac, duration, pred_hrz, occupancy, batch_id)
                if args.debug:
                    result = test_foundation_model(args, model, args.sampling_rate, hvac, duration, pred_hrz=pred_hrz, occupancy=occupancy, batch_id=batch_id, data_df=data_df)
                else:
                    try:
                        result = test_foundation_model(args, model, args.sampling_rate, hvac, duration, pred_hrz=pred_hrz, occupancy=occupancy, batch_id=batch_id, data_df=data_df)
                    except Exception as e:
                        print(f"Error in batch {batch_id}: {e}")
                        continue
                forecast = result["Prediction"]
                data = result["Data"]
                gt = result["GroundTruth"]
                directory = f'results/{args.model}_{args.real_data}'
                title = f'{args.pre_hrz}_{args.sampling_rate}_{args.duration}_{hvac}_{occupancy}_{batch_id}'
                if batch_id % 4 == 0:
                    plot_pred(data, forecast, gt=gt, _dir=directory, forecast_index=None, title=title)
                
                save_results_for_model(args.model, result['Data_with_timestamp'], result['Test_data'], forecast, duration, pred_hrz, hvac, occupancy, batch_id, directory+'_csv')


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="chronos", help="The type of models we are testing (chronos | moment)"
    )
    parser.add_argument(
        "--real_data", type=str, default="building", help="The type of data we are testing (building | electricity | electricity_uci | ecobee)"
    )
    parser.add_argument(
        "--debug", action='store_true'
    )
    args = parser.parse_args()
    main(args)
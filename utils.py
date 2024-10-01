import matplotlib.pyplot as plt  # requires: pip install matplotlib
import numpy as np
import pdb 
import pandas as pd
# from sklearn.metrics import mean_squared_error
import os
from get_data import get_real_building_data, get_electricity_data, \
                    get_uci_electricity_data, get_ecobee_temp_data, \
                    get_pecan_data, get_umass_data, get_elecdemand_data, \
                    get_subseasonal_data, get_pems04_data, \
                    get_loop_seattle_data, get_rlp_data, \
                    get_covid_data, get_c2000_data, \
                    get_restaurant_data, get_air_data
import warnings
try:
    import torch
except ImportError:
    print("Torch is not installed. Continuing without it.")


def plot_pred(org_data, median, _dir='None', gt=None, low=None, high=None, forecast_index=None, title=None):
    # print("Visualizing prediction data...")
    if forecast_index is None:
        forecast_index = range(len(org_data), len(org_data) + len(median))
    plt.figure(figsize=(8, 4))
    plt.plot(org_data, color="royalblue", label="historical data")
    plt.plot(forecast_index, median, color="tomato", label="median forecast")
    if low is not None and high is not None:
        plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
    if gt is not None:
        plt.plot(range(len(org_data), len(org_data) + len(gt)), gt, color="green", label="gt")
    plt.legend()
    plt.grid()
    directory = f"./{_dir}"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if title is not None:
        plt.savefig(f'{directory}/{title}.jpg')
    plt.close()

def signal_simulate(sampling_rate=100, duration=2, frequencies= [0.5, 5, 12], amplitudes= [1, 0.5, 0.2], pred_hrz=0.64):

    # Time vector
    t = np.linspace(0, duration+pred_hrz, int(sampling_rate * (duration+pred_hrz)), endpoint=False)

    # Generate the signal
    signal = np.zeros_like(t)
    for amplitude, frequency in zip(amplitudes, frequencies):
        signal += amplitude * np.sin(2 * np.pi * frequency * t)
    
    lookback = signal[:int(sampling_rate*duration)]
    gt = signal[int(sampling_rate*duration):]
    gt = gt[:int(sampling_rate*pred_hrz)]

    return lookback, gt

def calculate_metrics(gt, forecast):
    gt = np.float64(gt)
    if np.sum(np.isnan(forecast)) >= 1 or np.sum(np.isnan(gt)) >= 1:
        # pdb.set_trace()
        print('NaN detected...')
    # mse = mean_squared_error(gt, forecast)
    mse = np.mean((gt - forecast) ** 2)
    rmse = np.sqrt(mse)
    normalized_rmse = calc_normalized_rmse(gt, forecast)
    return mse, rmse, normalized_rmse

def save_results_for_model(model_name, data, test_data, forecast, duration, pred_hrz, mode, occupancy, batch_id, directory):
    """
    Save model predictions and evaluation metrics to a CSV file.

    Parameters:
    - model_name: str, name of the model
    - data: numpy array, input data used for prediction (what get_data functions returns)
    - test_data: numpy array, ground truth data for validation (what get_data functions returns)
    - forecast: numpy array, predictions made by the model
    - duration: int, duration of the data being analyzed
    - pred_hrz: int, prediction horizon
    - mode: str, mode of operation (e.g., 'off', 'heat')
    - occupancy: str, occupancy status ('occupied' or 'unoccupied')
    - batch_id: int, identifier for the current batch
    - directory: str, path to the directory where results should be saved

    Returns:
    None
    """

    mse, rmse, normalized_rmse = calculate_metrics(test_data[:, 0], forecast)
    
    value_data = data[:, 0]
    timestamp_data = data[:, -1]

    ground_truth = test_data[:, 0]
    timestamp_forecast = test_data[:, -1]

    result = {
        'batch_id': batch_id,
        'MSE': mse,
        'RMSE': rmse, 
        'Norm_RMSE': normalized_rmse, 
        'Data': value_data.tolist(),
        'Timestamp_data': timestamp_data.tolist(),
        'GroundTruth': ground_truth.tolist(),
        'Timestamp_forecast': timestamp_forecast.tolist(),
        'Forecast': forecast.tolist(),
        'Duration': duration,
        'PredictionHorizon': pred_hrz,
        'Mode': mode,
        'Occupancy': occupancy,
        'Model': model_name
    }

    # Load existing results if they exist
    file_path = f"{directory}/results_{mode}_{occupancy}/{duration}_{pred_hrz}/{model_name}.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['batch_id', 'MSE', 'RMSE', 'Norm_RMSE', 'Mode', 'Occupancy', 'Duration', 'PredictionHorizon', 'Data', 'Timestamp_data', 'GroundTruth', 'Timestamp_forecast', 'Forecast', 'Model'])

    # Append the new result
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    # Save the results back to the CSV file
    df.to_csv(file_path, index=False)

def analyze_time_series(ts: np.ndarray):
    from scipy.stats import linregress
    
    # Calculate the overall trend using linear regression
    x = np.arange(len(ts))

    slope, _, _, _, _ = linregress(x, ts)
    
    # Compute autocorrelation for different lags
    n = len(ts)
    ts_mean = ts - np.mean(ts)
    autocorr_full = np.correlate(ts_mean, ts_mean, mode='full') / (np.var(ts) * n+1e-8)
    autocorr = autocorr_full[n-1:]  # Keep the second half (non-negative lags)

    # Find the top lag (excluding lag 0)
    top_lag = np.argmax(autocorr[1:]) + 1  # +1 to account for zero-based indexing
    
    return slope, top_lag

dataset_descriptions = {
	'building': 'This dataset contains indoor temperature (in F) of an apartment in the US. (Sampling rate: {:.4f} hour).',
	'electricity': 'This dataset provides energy consumption data (in watts) from an apartment (Sampling rate: {:.4f} Hourly).',
    'ecobee': 'This dataset contains indoor temperature (in F) of an apartment in the US. (Sampling rate: {:.4f} hour).',
    'electricity_uci': 'This dataset provides energy consumption data from an apartment in watts (Sampling rate: {:.4f} Hourly).',
    'umass': 'This dataset provides energy consumption data from a single-family apartment in kW (Sampling rate: {:.4f} Hourly).',
    'elecdemand': 'This dataset contains electricity demand data (in GW) for Victoria, Australia. (Sampling rate: {:.4f} hour).',
    'subseasonal': 'This dataset contains temperature (in Celsius degree) for the western U.S. for subseasonal forecasting (Sampling rate: {:.4f} hour)',
    'pems04': 'This dataset contains traffic flow data collected from highway sensors in California, for monitoring and forecasting traffic conditions (Sampling rate: Every  {:.4f} hour).',
    'loop_seattle': 'The dataset consists of spatio-temporal speed data collected every 5 minutes by inductive loop detectors along Seattle area freeways, including I-5, I-405, I-90, and SR-520, with speed information averaged from multiple detectors at specific mileposts. (Sampling rate: Every  {:.4f} hour).',
    'rlp': 'This dataset provides energy consumption data from real-world customers in kW (Sampling rate: {:.4f} Hourly).',
    'covid': 'This dataset provides number of death due to covid-19 disease (Sampling rate: {:.4f} Hourly).',
    'c2000': 'The CMIP6 dataset provides climate model output data for the year 2000, used for studying climate change projections (Sampling rate: {:.4f} Hourly).',
    'restaurant': 'The dataset provides daily comsumer visit of a restaurant. (Sampling rate: {:.4f} Hourly).',
    'air': 'The time serie contains hourly air pollution data from multiple Chinese cities (Sampling rate: {:.4f} Hourly).',
}

def get_bert_embedding(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt")
    # Get hidden states from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # The hidden states are in the `last_hidden_state` tensor
    hidden_states = outputs.last_hidden_state

    # Typically, you might average the hidden states or take the CLS token's hidden state for a sentence embedding
    sentence_embedding = hidden_states.mean(dim=1)
    return sentence_embedding / sentence_embedding.norm()

def generate_emb_for_sample(args, data, sampling_rate, start_time, tokenizer, model):
    # dummy="The SZ_TAXI dataset records taxi trips in Shenzhen, China, helpful for traffic analysis and prediction (Sampling rate: Every {:.4f} minutes)."
    dummy="This is a dataset from an unknown source. (Sampling rate: Every {:.4f} minutes)."
    sr = sampling_rate / 3600
    ts = np.asarray(data, dtype=np.float32)
    template = "The input has a minimum of {:.2f}, a maximum of {:.2f}, and a median of {:.2f}. The overall trend is {:.2f}. The top lag is {:.2f}."
    timestamp_str = f'The collection time of the input starts at {start_time}. '
    slope, top_lag = analyze_time_series(ts)
    Min, Max, med = ts.min(), ts.max(), np.median(ts)
    if args.model in ('spacetime_lg_ctrl', 'spacetime_lg_v2_ctrl', 'attn_lg_v2_ctrl'):
        desc = dummy.format(sr)
    else:
        desc = dataset_descriptions[args.real_data].format(sr)
    sentence = desc + timestamp_str + template.format(Min, Max, med, slope, top_lag)
    emb = get_bert_embedding(sentence, tokenizer, model)
    return emb 

def test_foundation_model(args, model, sampling_rate, mode, duration, pred_hrz, occupancy, batch_id, data_df=None):
    
    # Simulate building data
    if args.real_data == 'building':
        data, test_data = get_real_building_data(duration, pred_hrz, sampling_rate, mode, occupancy=occupancy, batch_id=batch_id)
    elif args.real_data == 'electricity':
        data, test_data = get_electricity_data(duration, pred_hrz, sampling_rate, occupancy=occupancy, batch_id=batch_id)
    elif args.real_data == 'electricity_uci':
        data, test_data = get_uci_electricity_data(duration, pred_hrz, sampling_rate, house_id=occupancy, batch_id=batch_id, data_df=data_df)
    elif args.real_data == 'pecan':
        data, test_data = get_pecan_data(duration, pred_hrz, sampling_rate, house_id=occupancy, batch_id=batch_id, data_df=data_df)
    elif args.real_data == 'umass':
        data, test_data = get_umass_data(duration, pred_hrz, sampling_rate, house_id=occupancy, batch_id=batch_id, data_df=data_df)
    elif args.real_data == 'ecobee':
        data, test_data = get_ecobee_temp_data(duration, pred_hrz, sampling_rate, house_id=occupancy, batch_id=batch_id, data_df=data_df)
    elif args.real_data == 'elecdemand':
        data, test_data = get_elecdemand_data(duration, pred_hrz, sampling_rate, house_id=1, batch_id=batch_id, data_df=data_df)
    elif args.real_data == 'subseasonal':
        data, test_data = get_subseasonal_data(duration, pred_hrz, sampling_rate, house_id=1, batch_id=batch_id, data_df=data_df)
    elif args.real_data in ('pems04'):
        data, test_data = get_pems04_data(duration, pred_hrz, sampling_rate, house_id=1, batch_id=batch_id, data_df=data_df)
    elif args.real_data == 'loop_seattle':
        data, test_data = get_loop_seattle_data(duration, pred_hrz, sampling_rate, house_id=1, batch_id=batch_id, data_df=data_df)
    elif args.real_data == 'rlp':
        data, test_data = get_rlp_data(duration, pred_hrz, sampling_rate, house_id=1, batch_id=batch_id, data_df=data_df)
    elif args.real_data == 'covid':
        data, test_data = get_covid_data(duration, pred_hrz, sampling_rate, house_id=1, batch_id=batch_id, data_df=data_df)
    elif args.real_data == 'c2000':
        data, test_data = get_c2000_data(duration, pred_hrz, sampling_rate, house_id=1, batch_id=batch_id, data_df=data_df)
    elif args.real_data == 'restaurant':
        data, test_data = get_restaurant_data(duration, pred_hrz, sampling_rate, house_id=1, batch_id=batch_id, data_df=data_df)
    elif args.real_data == 'air':
        data, test_data = get_air_data(duration, pred_hrz, sampling_rate, house_id=1, batch_id=batch_id, data_df=data_df)
    gt = test_data[:, 0]
    pred_len = int(pred_hrz * 3600 / sampling_rate)
    #univariate data
    temp_data = data[:, 0]
    if args.model == 'moment':
        if temp_data.shape[0] + pred_len > 512:
            warnings.warn('Moment can at most handle 512 timestamp. Output will be trucated! \
Consider shorten your context length.') 
    if args.model == 'ARIMA':
        med = model(temp_data, pred_len)
    elif args.model == 'TimeGPT':
        med = model(data, sampling_rate, pred_len)
    elif args.model.lower() == 'lagllama':
        med = model(data, item_id='your_item_id', sampling_rate=sampling_rate, prediction_length=pred_len, normalize=False)
    elif args.model == 'Regression':
        med = model(data, test_data, prediction_length=pred_len)
    elif args.model == 'BestFitCurve' or args.model == 'Spline':
        med = model(data, test_data, sampling_rate)
    elif args.model == 'TimesFM':
        med = model(temp_data, prediction_length=pred_len)
    elif args.model in ('spacetime_ts', 'spacetime_ts_v2'):
        from datetime import datetime
        z = data[:, -1]
        ts = np.array([[
                (dt.year - 2000),  # Year offset by 2000
                dt.month,           # Month
                dt.day,             # Day
                dt.hour,            # Hour
                dt.minute           # Minute
            ] for dt in z]).astype(np.float32)
        low, med, high = model(temp_data, pred_len, ts=ts)
    elif args.model in ('attn_ts_v2', 'attn_pe_ts_v2'):
        from datetime import datetime
        z = data[:, -1]
        ts = np.array([[
                (dt.year - 2000),  # Year offset by 2000
                dt.month,           # Month
                dt.day,             # Day
                dt.hour,            # Hour
                dt.minute           # Minute
            ] for dt in z]).astype(np.float32)
        z_t = test_data[:, -1]
        ts_t = np.array([[
                (dt.year - 2000),  # Year offset by 2000
                dt.month,           # Month
                dt.day,             # Day
                dt.hour,            # Hour
                dt.minute           # Minute
            ] for dt in z_t]).astype(np.float32)
        low, med, high = model(temp_data, pred_len, ts=ts, ts_t=ts_t)
    elif args.model in ('spacetime_lg', 'spacetime_lg_v2', 'spacetime_lg_ctrl', 'spacetime_lg_v2_ctrl', 'attn_lg_v2', 'attn_lg_v2_ctrl'):
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        bert_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        lg = generate_emb_for_sample(args, temp_data, sampling_rate, data[0, -1], tokenizer, bert_model)
        # pdb.set_trace()
        low, med, high = model(temp_data, pred_len, lg=lg)
    else:
        low, med, high = model(temp_data, pred_len)
    # Return predictions
    return {
        'Prediction': med,
        'GroundTruth': gt,
        'Data':temp_data,
        'Data_with_timestamp': data,
        'Test_data': test_data
    }

def load_foundation_model(args, pred_hrz):

    if args.model == "chronos":
        from models import ChronosModel
        model = ChronosModel(name = "amazon/chronos-t5-large",
                            device = "cuda")
    elif args.model == "moment":
        from models import MomentModels
        pred_len = int(pred_hrz * 3600 / args.sampling_rate)
        model = MomentModels(name = "AutonLab/MOMENT-1-large",
                            prediction_length = pred_len)
    elif args.model == 'ARIMA':
        from models import ARIMAModel
        model = ARIMAModel(p=5, d=1, q=0)
    elif args.model == 'AutoARIMA':
        from models import AutoARIMAModel
        model = AutoARIMAModel()
    elif args.model == 'SeasonalARIMA':
        from models import SeasonalAutoARIMAModel
        model = SeasonalAutoARIMAModel()
    elif args.model == 'TimeGPT':
        from models import TimeGPTModel
        model = TimeGPTModel()
    elif args.model == 'LagLlama':
        from lag_src.lagllama_model import LagLlamaModel
        model = LagLlamaModel()
    elif args.model == 'Regression':
        from models import RegressionModel
        model = RegressionModel()
    elif args.model == 'BestFitCurve':
        from models import BestFitCurveModel
        model = BestFitCurveModel()
    elif args.model == 'Spline':
        from models import SplineModel
        model = SplineModel()
    elif args.model == 'TimesFM':
        from models import TimesFMModel
        model = TimesFMModel()
    elif args.model == 'uni2ts':
        from models import Uni2TSModel
        pred_len = int(pred_hrz * 3600 / args.sampling_rate)
        context_len = int(args.duration * 3600 / args.sampling_rate)
        model = Uni2TSModel(prediction_length=pred_len, context_length=context_len)
    elif args.model in ('spacetime', 'spacetime_ts', 'spacetime_lg',\
            'spacetime_ts_v2', 'spacetime_lg_v2', 'spacetime_lg_ctrl', 'spacetime_lg_v2_ctrl'):
        from models import MambaTSFM
        pred_len = int(pred_hrz * 3600 / args.sampling_rate)
        context_len = int(args.duration * 3600 / args.sampling_rate)
        model = MambaTSFM(context_len, pred_len, model=args.model)
    elif args.model in ('attn', 'attn_pe', 'attn_lg_v2', 'attn_ts_v2', 'attn_pe_ts_v2', 'attn_lg_v2_ctrl'):
        from models import ATTNTSFM
        pred_len = int(pred_hrz * 3600 / args.sampling_rate)
        context_len = int(args.duration * 3600 / args.sampling_rate)
        model = ATTNTSFM(context_len, pred_len, model_name=args.model)
    else:
        raise NotImplementedError("Model hasn't been incorporated!")
    
    return model 

def calc_normalized_rmse(y_true, y_pred):
    """
    Calculate the normalized Root Mean Square Error (RMSE) between two arrays.

    Parameters:
    y_true (array-like): The ground truth values.
    y_pred (array-like): The predicted values.

    Returns:
    float: The normalized RMSE.
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Normalize RMSE
    range_of_data = np.max(y_true) - np.min(y_true)
    normalized_rmse = rmse / (range_of_data+1e-20)
    
    return normalized_rmse

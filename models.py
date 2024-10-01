import torch
import numpy as np 
import os 
import pdb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
import matplotlib.patheffects as pe
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


class ChronosModel:
    
    def __init__(self, name, device="cuda"):
        from chronos import ChronosPipeline
        self.model = ChronosPipeline.from_pretrained(
            name,
            device_map=device,  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )
        self.scaler = MinMaxScaler()

    def __call__(self, data, prediction_length, num_samples=1):
        # Normalizing the data
        data = np.array(data).reshape(-1, 1)  # Reshape data for the scaler
        normalized_data = self.scaler.fit_transform(data)

        if not torch.is_tensor(normalized_data):
            _data = torch.tensor(normalized_data.flatten(), dtype=torch.float32)  # Flatten for the model
        else:
            _data = normalized_data

        forecast = self.model.predict(
            context=_data,
            prediction_length=prediction_length,
            num_samples=num_samples,
            limit_prediction_length=False,
        )

        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

        # Denormalizing the predictions
        low = self.scaler.inverse_transform(low.reshape(-1, 1)).flatten()
        median = self.scaler.inverse_transform(median.reshape(-1, 1)).flatten()
        high = self.scaler.inverse_transform(high.reshape(-1, 1)).flatten()

        return low, median, high  # 80% interval


class MomentModels:
    
    def __init__(self, name, prediction_length):
        from momentfm import MOMENTPipeline
        self.model = MOMENTPipeline.from_pretrained(
            name, 
            model_kwargs={'task_name': 'reconstruction'},
        )
        self.model.init()
        self.scaler = MinMaxScaler()
    
    def __call__(self, data, prediction_length=None):
        # Normalizing the data
        data = np.array(data).reshape(-1, 1)  # Reshape data for the scaler
        normalized_data = self.scaler.fit_transform(data)
        normalized_data = normalized_data.flatten()

        if not torch.is_tensor(normalized_data):
            _data = torch.tensor(normalized_data).float()
        else:
            _data = normalized_data.float()
        
        if len(_data.shape) != 3:
            print("Appending extra dimensions to batch and dim")
            _data = _data[None, None, :]
        
        if _data.shape[-1] <= 512:
            length = _data.shape[-1]
            print("Left appending zeros to the data")
            data_pad = torch.cat([_data, torch.zeros(1, 1, 512-length)], 2)
            int_mask = torch.cat([torch.ones(1, length), torch.zeros(1, 512-length)], 1)
            output = self.model(data_pad, input_mask=int_mask, mask=int_mask)
        else:
            return None, np.array([np.nan]*prediction_length), None
            output = self.model(_data)
            

        output = output.reconstruction.detach().squeeze()
        output = output[int_mask[0]==0][: prediction_length]
        
        # Denormalizing the predictions
        output = output.reshape(-1, 1)  # Reshape for the scaler
        denormalized_output = self.scaler.inverse_transform(output).flatten()

        return None, denormalized_output, None




class TimeGPTModel:
    # pip install nixtla>=0.5.1
    def __init__(self):
        from nixtla import NixtlaClient
        # 1. Instantiate the NixtlaClient
        self.model = NixtlaClient(api_key="nixtla-tok-yactP8b57EdAZTXX7NC5Qen2sw8QqVm0WaW1Xmnebl8KWnt5eTVnH634OUbDoYyEVbmhqHHZ4khqfPMg")
        self.scaler = MinMaxScaler()

    def __call__(self, data, sampling_rate, prediction_length=None, long_term=False, normalize=False):
        # Separate the values and timestamps
        values = data[:, 0]
        timestamps = data[:, -1]

        # Print shapes for debugging
        #print(f"Values shape: {values.shape}")
        #print(f"Timestamps shape: {timestamps.shape}")

        # Normalize the values
        values = np.array(values).reshape(-1, 1)
        if normalize:
            normalized_values = self.scaler.fit_transform(values).flatten()
        else:
            normalized_values = values.flatten()

        # Convert numpy array to pandas dataframe using the provided timestamps
        timestamps = pd.to_datetime(timestamps)

        # Ensure the lengths match
        if len(normalized_values) != len(timestamps):
            raise ValueError("Mismatch in lengths of values and timestamps")

        # Check if timestamps are consecutive
        expected_interval = pd.Timedelta(seconds=sampling_rate)
        actual_intervals = timestamps.diff().dropna()
        if not (actual_intervals == expected_interval).all():
            raise ValueError("Timestamps are not consecutive")

        df = pd.DataFrame({'value': normalized_values, 'timestamp': timestamps})
        df.set_index('timestamp', inplace=True)

        # Address NaN values using linear interpolation
        df['value'] = df['value'].interpolate(method='linear')

        # Drop any rows with NaN values after interpolation
        df = df.dropna()

        #print(f"Shape of df after interpolation and dropping NaNs: {df.shape}")
        #print(df.head())

        # Ensure the DataFrame has the required columns and index name
        df.reset_index(inplace=True)
        df.columns = ['timestamp', 'value']

        # Ensure no NaNs in the final DataFrame
        if df.isnull().values.any():
            raise ValueError("DataFrame contains NaN values")

        #print(f"DataFrame before forecast: {df.head()}")

        # Validate the DataFrame structure
        if not all([col in df.columns for col in ['timestamp', 'value']]):
            raise ValueError("DataFrame does not contain the required columns")

        # Validate the lengths of the columns
        if len(df['timestamp']) != len(df['value']):
            raise ValueError("Mismatch in lengths of DataFrame columns")

        #print(f"DataFrame length check: {len(df['timestamp']) == len(df['value'])}")

        # Assign the correct frequency based on sampling_rate
        freq = self.get_freq_alias(sampling_rate)
        #print(f"Assigned frequency: {freq}")

        # Forecast the next prediction_length steps
        if not long_term:
            fcst_df = self.model.forecast(df, h=prediction_length, time_col='timestamp', freq=freq, target_col='value')
        else:
            fcst_df = self.model.forecast(df, h=prediction_length, time_col='timestamp', target_col='value', freq=freq, model='timegpt-1-long-horizon')

        # Extract the forecasted normalized values
        normalized_forecast = fcst_df['TimeGPT'].values.reshape(-1, 1)

        if normalize:
            # Denormalize the forecasted values
            denormalized_forecast = self.scaler.inverse_transform(normalized_forecast).flatten()
        else:
            denormalized_forecast = normalized_forecast.flatten()

        return denormalized_forecast

    def get_freq_alias(self, sampling_rate):
        if sampling_rate < 60:
            return f'{sampling_rate}S'
        elif sampling_rate < 3600:
            return f'{sampling_rate // 60}T'
        elif sampling_rate < 86400:
            return f'{sampling_rate // 3600}H'
        else:
            return f'{sampling_rate // 86400}D'




class TimesFMModel:
    def __init__(self, device="cuda"):
        import timesfm

        backend = 'gpu' if device == "cuda" else 'cpu'
        self.model = timesfm.TimesFm(
            context_len=512,
            horizon_len=192,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend=backend,
        )
        self.model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
        self.scaler = MinMaxScaler()

    def __call__(self, data, freq_level=0, prediction_length=None):
        '''
        freq (int): choose from 0, 1, or 2, where
            0 (default): high frequency, long horizon time series. We recommend using this for time series up to daily granularity.
            1: medium frequency time series. We recommend using this for weekly and monthly data.
            2: low frequency, short horizon time series. We recommend using this for anything beyond monthly, e.g. quarterly or yearly.
        '''
        # Normalize the data
        data = np.array(data).reshape(-1, 1)
        normalized_data = self.scaler.fit_transform(data).flatten()

        forecast_input = [normalized_data] # Expect tuples (list) as input, i.e., [(seq_len,)]
        freq = [freq_level] # Expect tuples (list) as input
        point_forecast, experimental_quantile_forecast = self.model.forecast(forecast_input, freq)
        point_forecast = np.array(point_forecast[0]).reshape(-1, 1)

        # Denormalize the forecasted values
        denormalized_forecast = self.scaler.inverse_transform(point_forecast).flatten()

        return denormalized_forecast[:prediction_length]

import sys
import yaml
def load_config_to_args(args, config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    for key, value in config.items():
        setattr(args, key, value)
    
    return args

def modify_key(state_dict):
    new_state_dict = {}
    for k in state_dict:
        if 'module.' in k:
            new_key = k.split('module.')[1]
            new_state_dict[new_key] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]
    return new_state_dict

def return_my_tsfm(
    model_name, context_len, pred_len, input_transform=None, output_transform=None, device='cuda'
):  
    module_path = "/home/prquan/Github/TS-foundation-model/Mamba-TSFM"
    if module_path not in sys.path:
        sys.path.append(module_path)
    from omegaconf import OmegaConf 
    from setup import load_model_config, load_main_config, initialize_args
    from setup.configs.model import update_output_config_from_args  # For multivariate feature prediction
    from model.network import SpaceTime
    from model.attn import GPT
    from data_transforms import get_data_transforms
    import yaml

    _args = initialize_args()
    if model_name == 'spacetime':
        config_file = f'{module_path}/config.yaml'
        # load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=spacetime-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=64-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
        load_model_path = f'{module_path}/checkpoints/ckp_3090/bval-m=spacetime-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=64-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
    elif model_name == 'spacetime_ts':
        config_file = f'{module_path}/config_ts_small.yaml'
        load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=spacetime_ts-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=64-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
    elif model_name in ('spacetime_lg', 'spacetime_lg_ctrl'):
        config_file = f'{module_path}/config_lg_small.yaml'
        load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=spacetime_lg-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=64-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
    elif model_name in ('spacetime_lg_v2', 'spacetime_lg_v2_ctrl'):
        config_file = f'{module_path}/config_lg_small_v2.yaml'
        load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=spacetime_lg_v2-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=64-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
        # load_model_path = f'{module_path}/checkpoints/customized_a/bval-m=spacetime_lg_v2-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=64-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
    elif model_name in ('spacetime_ts_v2'):
        config_file = f'{module_path}/config_ts_small_v2.yaml'
        # load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=spacetime_ts_v2-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=64-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
        load_model_path = f'{module_path}/checkpoints/ckp_3090/bval-m=spacetime_ts_v2-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=64-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
    elif model_name == 'attn':
        config_file = f'{module_path}/attn.yaml'
        load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=attn-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=32-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=5-re=0-se=0.pth'
    elif model_name == 'attn_pe':
        config_file = f'{module_path}/attn_pe.yaml'
        load_model_path = f'{module_path}/checkpoints/customized_a/bval-m=attn_pe-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=32-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=5-re=0-se=0.pth'
    elif model_name == 'attn_pe_ts_v2':
        config_file = f'{module_path}/attn_pe_ts_small_v2.yaml'
        load_model_path = f'{module_path}/checkpoints/customized_a/bval-m=attn_pe_ts_v2-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=32-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=5-re=0-se=0.pth'
    elif model_name in ('attn_lg_v2', 'attn_lg_v2_ctrl'):
        config_file = f'{module_path}/attn.yaml'
        load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=attn_lg_v2-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=32-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=5-re=0-se=0.pth'
    elif model_name == 'attn_ts_v2':
        config_file = f'{module_path}/attn.yaml'
        # load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=attn_ts_v2-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=32-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
        load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=attn_ts_v2-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=32-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=128-vm=ir-me=500-ese=20-re=0-se=0.pth'
    with open(config_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
    load_config_to_args(_args, config_file)

    _args.lag = context_len
    _args.horizon = pred_len
    
    # Initialize Model
    model_configs = {'embedding_config': _args.embedding_config,
                    'encoder_config':   _args.encoder_config,
                    'decoder_config':   _args.decoder_config,
                    'output_config':    _args.output_config}
    model_configs = OmegaConf.create(model_configs)
    model_configs = load_model_config(model_configs, config_dir=f'{module_path}/configs/model',
                                    args=_args)
    
    
    model_configs['inference_only'] = False
    model_configs['lag'] = _args.lag
    model_configs['horizon'] = _args.horizon

    # if model_name == 'spacetime_ts':
    #     model_configs['use_ts'] = True
    # if model_name in ('spacetime_lg', 'spacetime_lg_ctrl'):
    #     model_configs['use_lg'] = True
    use_ts, use_lg, use_pe = False, False, False
    if '_ts_v2' in model_name:
        use_ts = 'v2'
        model_configs['use_ts'] = use_ts
    elif '_ts' in model_name:
        use_ts = 'v1'
        model_configs['use_ts'] = use_ts

    if '_lg_v2' in model_name:
        use_lg = 'v2' 
        model_configs['use_lg'] = use_lg
    elif '_lg' in model_name:
        use_lg = 'v1' 
        model_configs['use_lg'] = use_lg
    
    if '_pe' in model_name:
        use_pe = True
        model_configs['use_pe'] = use_pe

    if model_name in ('spacetime', 'spacetime_ts', 'spacetime_ts_v2', 'spacetime_lg', \
                      'spacetime_lg_v2', 'spacetime_lg_ctrl', 'spacetime_lg_v2_ctrl'):
        model = SpaceTime(**model_configs)
        # load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=spacetime-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=64-ki=no-no=1-la=192-ho=48-fe=S-dt=mean-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
        # load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=spacetime-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=64-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
        model.replicate = _args.replicate  # Only used for testing specific things indicated by replicate
        model.set_lag(_args.lag)
        model.set_horizon(_args.horizon)
    elif 'attn' in model_name:
        # model = GPT(lag=_args.lag, horizon=_args.horizon)
        model = GPT(lag=_args.lag, horizon=_args.horizon, d=_args.kernel_dim, use_ts=use_ts, use_lg=use_lg, use_pe=use_pe)
        # load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=attn-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=64-ki=no-no=1-la=192-ho=48-fe=S-dt=mean-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
        # load_model_path = f'{module_path}/checkpoints/customized_a/btrn-m=attn-ec=repeat-pc=default-ec=default-dc=default-oc=default-nb=1-nk=32-nh=4-kd=64-ki=no-no=1-la=192-ho=48-fe=S-dt=standardize-cw=1+1+1-lo=ir-dr=0.25-lr=0.001-op=adamw-sc=tc-wd=0.0001-bs=64-vm=ir-me=500-ese=20-re=0-se=0.pth'
    else:
        raise NotImplementedError

    dictionary = torch.load(load_model_path, map_location=device)
    modified_state_key = modify_key(dictionary['state_dict'])
    model.load_state_dict(modified_state_key)
    print(f'Model loaded from {load_model_path}')

    model = model.to(device)

    input_transform, output_transform = get_data_transforms(_args.data_transform,
                                                        _args.lag)

    if input_transform is None:
        input_transform = lambda x : x 
    if output_transform is None:
        output_transform = lambda x : x 

    return model, input_transform, output_transform

class MambaTSFM:
    def __init__(self, context_len, pred_len, model, input_transform=None, output_transform=None, device='cuda'):
        self.device = device 
        self.model_name = model
        model, input_transform, output_transform = return_my_tsfm(self.model_name, context_len, pred_len, input_transform, output_transform, device)
        self.model = model 
        self.input_transform, self.output_transform = input_transform, output_transform

    def __call__(self, data, prediction_length, ts=None, lg=None):
        
        data = np.float32(data[None,:,None])

        data = torch.from_numpy(data)
        u = self.input_transform(data)
        u = u.to(self.device)

        if '_ts' in self.model_name:
            ts = np.float32(ts[None,:,:])
            ts = torch.from_numpy(ts)
            ts = ts.to(self.device)
            y_pred, z_pred = self.model(u, ts=ts) 
        elif '_lg' in self.model_name:
            lg = lg.to(self.device)
            y_pred, z_pred = self.model(u, lg=lg) 
        else:
            y_pred, z_pred = self.model(u)
        y_pred = [self.output_transform(_y) if _y is not None else _y 
                      for _y in y_pred]
        y_c, y_o = y_pred 
        y_c = y_c.squeeze()
        y_c = y_c.cpu().detach().numpy()
        return None, y_c, None

class ATTNTSFM:
    def __init__(self, context_len, pred_len, model_name, input_transform=None, output_transform=None, device='cuda'):
        self.device = device 
        self.model_name = model_name
        model, input_transform, output_transform = return_my_tsfm(self.model_name, context_len, pred_len, input_transform, output_transform, device)
        self.model = model 
        self.context_len, self.pred_len = context_len, pred_len
        self.input_transform, self.output_transform = input_transform, output_transform

    def __call__(self, data, pred_len=None, ts=None, ts_t=None, lg=None):
        
        data = np.float32(data[None,:,None])

        data = torch.from_numpy(data)
        u = self.input_transform(data)
        u = u.to(self.device)
        
        if '_ts' in self.model_name:
            ts = np.float32(ts[None,:,:])
            ts = torch.from_numpy(ts)
            ts = ts.to(self.device)

            ts_t = np.float32(ts_t[None,:,:])
            ts_t = torch.from_numpy(ts_t)
            ts_t = ts_t.to(self.device)
            
        elif '_lg' in self.model_name:
            lg = lg.to(self.device)

        max_new_tokens = pred_len if pred_len is not None else self.pred_len
        y_pred, z_pred = self.model.generate(u, max_new_tokens, ts=ts, ts_t=ts_t, lg=lg)
        y_pred = [self.output_transform(_y) if _y is not None else _y 
                      for _y in y_pred]
        y_c = y_pred[0]
        y_c = y_c.cpu().squeeze().detach().numpy()
        return None, y_c, None
        


class ARIMAModel:
    
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q
        self.scaler = MinMaxScaler()

    def __call__(self, data, prediction_length):
        from statsmodels.tsa.arima.model import ARIMA
        # Normalizing the data
        data = np.array(data).reshape(-1, 1)  # Reshape data for the scaler
        normalized_data = self.scaler.fit_transform(data)
        
        # Flatten the normalized data for ARIMA
        flattened_data = normalized_data.flatten()

        # Fit the ARIMA model
        model = ARIMA(flattened_data, order=(self.p, self.d, self.q))
        model_fit = model.fit()

        # Make predictions
        forecast = model_fit.forecast(steps=prediction_length)

        # Denormalizing the predictions
        forecast = forecast.reshape(-1, 1)
        forecast = self.scaler.inverse_transform(forecast).flatten()

        return forecast

class AutoARIMAModel:

    def __init__(self):
        self.scaler = MinMaxScaler()

    def __call__(self, data, prediction_length):
        import pmdarima as pm
        # Normalizing the data
        data = np.array(data).reshape(-1, 1)  # Reshape data for the scaler
        normalized_data = self.scaler.fit_transform(data)
        
        # Flatten the normalized data for ARIMA
        flattened_data = normalized_data.flatten()

        # Handle NaN values using linear interpolation
        nans, x = np.isnan(flattened_data), lambda z: z.nonzero()[0]
        if np.any(nans):
            flattened_data[nans] = np.interp(x(nans), x(~nans), flattened_data[~nans])

        # Fit the auto_arima model
        model = pm.auto_arima(flattened_data, 
                              start_p=1, start_q=1,
                              test='adf',       # use adftest to find optimal 'd'
                              max_p=5, max_q=5, # maximum p and q
                              m=1,              # frequency of series
                              d=None,           # let model determine 'd'
                              seasonal=False,   # No Seasonality
                              start_P=0, 
                              D=0, 
                              trace=True,
                              error_action='ignore',  
                              suppress_warnings=True, 
                              stepwise=True)

        # Make predictions
        forecast = model.predict(n_periods=prediction_length)

        # Denormalizing the predictions
        forecast = np.array(forecast).reshape(-1, 1)
        forecast = self.scaler.inverse_transform(forecast).flatten()

        return None, forecast, None

    def plot_forecast(self, data, forecast, test_data, sampling_rate):
        # Prepare the time array for data
        t_data = np.arange(len(data)) / (3600 / sampling_rate)
        data_len = len(data)
        
        # Prepare the time array for test_data
        t_test = np.arange(len(test_data)) / (3600 / sampling_rate) + t_data[-1]

        # Plot the data
        plt.figure(figsize=(12, 6))
        t_extended = np.concatenate((t_data, t_test))
        plt.plot(t_data, data, label='Data', color='blue')
        plt.plot(t_test, forecast, label='Forecast', color='orange')
        plt.plot(t_test, test_data, label='Ground Truth', color='green', linestyle='dashed')

        # Add annotations
        plt.title('ARIMA Forecasting')
        plt.xlabel('Time (hours)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.show()

class SeasonalAutoARIMAModel:
    def __init__(self, max_p=3, max_q=3):
        self.scaler = MinMaxScaler()
        self.max_p, self.max_q = max_p, max_q

    def __call__(self, data, prediction_length, seasonal_period=12):
        import pmdarima as pm
        # Normalizing the data
        data = np.array(data).reshape(-1, 1)  # Reshape data for the scaler
        normalized_data = self.scaler.fit_transform(data)
        
        # Flatten the normalized data for ARIMA
        flattened_data = normalized_data.flatten()

        # Handle NaN values using linear interpolation
        nans, x = np.isnan(flattened_data), lambda z: z.nonzero()[0]
        if np.any(nans):
            flattened_data[nans] = np.interp(x(nans), x(~nans), flattened_data[~nans])

        # Fit the auto_arima model
        model = pm.auto_arima(flattened_data, 
                              start_p=1, start_q=1,
                              test='adf',       # use adftest to find optimal 'd'
                              max_p=self.max_p, max_q=self.max_q, # maximum p and q
                              m=seasonal_period, # frequency of series for seasonality
                              d=None,           # let model determine 'd'
                              seasonal=True,    # Seasonality enabled
                              start_P=0, 
                              D=1,              # Seasonal differencing
                              trace=True,
                              error_action='ignore',  
                              suppress_warnings=True, 
                              stepwise=True)

        # Make predictions
        forecast = model.predict(n_periods=prediction_length)

        # Denormalizing the predictions
        forecast = np.array(forecast).reshape(-1, 1)
        forecast = self.scaler.inverse_transform(forecast).flatten()

        return None, forecast, None

class RegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def __call__(self, data, test_data, prediction_length):
        data=data[:,:-1]
        test_data=test_data[:,:-1]

        data = np.array(data)
        X_train = data[:-1]  # Use all rows except the last one for training
        y_train = data[1:, 0]  # Use the next step of the first column as ground truth

        # Train the regression model
        self.model.fit(X_train, y_train)
        
        # Make step-by-step predictions using the test data
        test_data = np.array(test_data)
        forecast = []
        
        for i in range(prediction_length):
            # Predict the next step using the current test data row
            if i == 0:
                next_pred = self.model.predict([test_data[i]])[0]
            else:
                features = np.hstack(([next_pred], test_data[i, 1:]))
                #print('features:',features)
                next_pred = self.model.predict([features])[0]

            forecast.append(next_pred)
            
        return np.array(forecast)


    def plot_forecast(self, data, forecast, test_data, sampling_rate):
        # Prepare the time array for data
        t_data = np.arange(len(data)) / (3600 / sampling_rate)
        data_len = len(data)
        
        # Prepare the time array for test_data
        t_test = np.arange(len(test_data)) / (3600 / sampling_rate) + t_data[-1]

        # Plot the data
        plt.figure(figsize=(12, 6))
        t_extended = np.concatenate((t_data, t_test))
        plt.plot(t_data, data[:, 0], label='Data', color='blue')
        plt.plot(t_test, forecast, label='Forecast', color='orange')
        plt.plot(t_test, test_data[:, 0], label='Ground Truth', color='green', linestyle='dashed')

        # Add annotations
        plt.title('Linear Regression Forecasting')
        plt.xlabel('Time (hours)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class UnivariateRegression:
    def __init__(self):
        self.model = LinearRegression()

    def _extract_features(self, timestamps):
        # Extract time of day and day of week from timestamps
        timestamps = pd.to_datetime(timestamps)
        time_of_day = timestamps.hour + timestamps.minute / 60.0
        day_of_week = timestamps.dayofweek

        # Combine features into a single DataFrame
        features = pd.DataFrame({
            'time_of_day': time_of_day,
            'day_of_week': day_of_week
        })

        return features

    def __call__(self, data, test_data, prediction_length):
        data = np.array(data)
        test_data = np.array(test_data)

        # Extract features from timestamps
        data_features = self._extract_features(data[:, -1])
        test_data_features = self._extract_features(test_data[:, -1])

        # Combine data features and test features to ensure consistent dummy variable creation
        combined_features = pd.concat([data_features, test_data_features])

        # Create dummy variables
        combined_features = pd.get_dummies(combined_features, columns=['time_of_day', 'day_of_week'])

        # Split back into data and test features
        train_features = combined_features.iloc[:len(data_features)]
        test_features = combined_features.iloc[len(data_features):]

        # Combine temperature and extracted features for training
        X_train = np.hstack([data[:-1, [0]], train_features.iloc[:-1].values])  # Use all rows except the last one for training
        y_train = data[1:, 0]  # Use the next step of the first column as ground truth

        # Train the regression model
        self.model.fit(X_train, y_train)

        # Make step-by-step predictions using the test data
        forecast = []
        current_temp = data[-1, 0]

        for i in range(prediction_length):
            # Combine current temperature and extracted features for prediction
            features = np.hstack(([current_temp], test_features.iloc[i].values))
            next_pred = self.model.predict([features])[0]
            forecast.append(next_pred)
            current_temp = next_pred

        return np.array(forecast)

    def plot_forecast(self, data, forecast, test_data, sampling_rate):
        # Prepare the time array for data
        t_data = np.arange(len(data)) / (3600 / sampling_rate)
        data_len = len(data)

        # Prepare the time array for test_data
        t_test = np.arange(len(test_data)) / (3600 / sampling_rate) + t_data[-1]

        # Plot the data
        plt.figure(figsize=(12, 6))
        t_extended = np.concatenate((t_data, t_test))
        plt.plot(t_data, data[:, 0], label='Data', color='blue')
        plt.plot(t_test, forecast, label='Forecast', color='orange')
        plt.plot(t_test, test_data[:, 0], label='Ground Truth', color='green', linestyle='dashed')

        # Add annotations
        plt.title('Linear Regression Forecasting')
        plt.xlabel('Time (hours)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_forecast(self, data, forecast, test_data, sampling_rate):
        # Prepare the time array for data
        t_data = np.arange(len(data)) / (3600 / sampling_rate)
        data_len = len(data)
        
        # Prepare the time array for test_data
        t_test = np.arange(len(test_data)) / (3600 / sampling_rate) + t_data[-1]

        # Plot the data
        plt.figure(figsize=(12, 6))
        t_extended = np.concatenate((t_data, t_test))
        plt.plot(t_data, data[:, 0], label='Data', color='blue')
        plt.plot(t_test, forecast, label='Forecast', color='orange')
        plt.plot(t_test, test_data[:, 0], label='Ground Truth', color='green', linestyle='dashed')

        # Add annotations
        plt.title('Linear Regression Forecasting')
        plt.xlabel('Time (hours)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.show()


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class DecayCurveModel:
    def __init__(self):
        self.params = None

    @staticmethod
    def exponential_func(t, T_0, time_constant):
        return T_0 * np.exp(-t / time_constant)

    def __call__(self, data, test_data, sampling_rate):
        data = np.array(data)
        test_data = np.array(test_data)
        
        # Prepare the time array for training data
        t = np.arange(len(data)) / (3600 / sampling_rate)
        
        # Temperature data
        T_t = data[:, 0]

        # Perform curve fitting with initial guess for T_0 and time_constant
        initial_guess = [T_t[0], 1]
        self.params, _ = curve_fit(
            self.exponential_func,
            t,
            T_t,
            p0=initial_guess
        )
        
        # Prepare the time array for test_data
        t_pred = np.arange(len(test_data)) / (3600 / sampling_rate) + t[-1]
        
        # Generate the fitted curve for the test period
        forecast = self.exponential_func(t_pred - t[-1], *self.params)

        # Plot the data and the fitted curve
        plt.figure(figsize=(12, 6))
        plt.plot(t, T_t, label='Data', color='blue')
        plt.plot(t_pred, forecast, label='Forecast', color='orange')
        plt.plot(t_pred, test_data[:, 0], label='Ground Truth', color='green', linestyle='dashed')
        plt.title('Exponential Decay Curve Fitting and Forecast')
        plt.xlabel('Time (hours)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.show()

        return forecast

class BestFitCurveModel:
    def __init__(self):
        self.params = None
        self.func = None

    @staticmethod
    def polynomial_func(t, a, b, c, d, e, f):
        # Example of a 5th degree polynomial
        return a * t**5 + b * t**4 + c * t**3 + d * t**2 + e * t + f

    def __call__(self, data, test_data, sampling_rate, func=None):
        if func is None:
            func = self.polynomial_func  # Use polynomial as default

        self.func = func
        data = np.array(data)
        test_data = np.array(test_data)
        
        # Prepare the time array for training data
        t = np.arange(len(data)) / (3600 / sampling_rate)
        
        # Temperature data
        T_t = data[:, 0]

        # Perform curve fitting with an initial guess
        initial_guess = np.ones(len(func.__code__.co_varnames) - 1)  # Simple initial guess
        self.params, _ = curve_fit(
            self.func,
            t,
            T_t,
            p0=initial_guess
        )
        
        # Prepare the time array for test_data
        t_pred = np.arange(len(test_data)) / (3600 / sampling_rate) + t[-1]
        
        # Generate the fitted curve for the test period
        forecast = self.func(t_pred - t[-1], *self.params)

        """
        # Plot the data and the fitted curve
        plt.figure(figsize=(12, 6))
        plt.plot(t, T_t, label='Data', color='blue')
        plt.plot(t_pred, forecast, label='Forecast', color='orange')
        plt.plot(t_pred, test_data[:, 0], label='Ground Truth', color='green', linestyle='dashed')
        plt.title('Best Fit Curve and Forecast')
        plt.xlabel('Time (hours)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.show()
        """
        return forecast




class SplineModel:
    def __init__(self, smoothing_factors=[0.1, 0.5, 1, 5, 10]):
        self.smoothing_factors = smoothing_factors
        self.best_smoothing_factor = None
        self.spline = None

    def fit(self, data, sampling_rate):
        data = np.array(data)
        
        # Prepare the time array for data
        t_data = np.arange(len(data)) / (3600 / sampling_rate)

        # Initial temperature
        T_t = data[:, 0]

        # Try different smoothing factors and select the best one based on training error
        best_mse = float('inf')
        
        for s in self.smoothing_factors:
            spline = UnivariateSpline(t_data, T_t, s=s)
            train_forecast = spline(t_data)
            mse = mean_squared_error(T_t, train_forecast)
            if mse < best_mse:
                best_mse = mse
                self.best_smoothing_factor = s
                self.spline = spline

    def __call__(self, data, test_data, sampling_rate):
        # Fit the model with the best smoothing factor
        self.fit(data, sampling_rate)
        
        data = np.array(data)
        test_data = np.array(test_data)

        # Prepare the time array for data
        t_data = np.arange(len(data)) / (3600 / sampling_rate)
        
        # Prepare the time array for test_data
        t_test = np.arange(len(test_data)) / (3600 / sampling_rate) + t_data[-1]

        # Make predictions using the spline function
        forecast = self.spline(t_test)

        return np.array(forecast)

class Uni2TSModel:

    def __init__(self, prediction_length, context_length, size="large", patch_size="auto", device="cuda"):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        self.size = size
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.patch_size = patch_size
        self.device = device
        
        self.model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{self.size}"),
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            # patch_size=self.patch_size,
            patch_size=32,
            num_samples=20,
            target_dim=1,
            feat_dynamic_real_dim=None,  # Set these dynamically
            past_feat_dynamic_real_dim=None,  # Set these dynamically
        )

    def __call__(self, data, num_samples=20):
        from einops import rearrange
        # Convert data to GluonTS dataset
        # Time series values. Shape: (batch, time, variate)
        data = np.float64(data)
        # Handle NaN values using linear interpolation
        nans, x = np.isnan(data), lambda z: z.nonzero()[0]
        if np.all(nans):
            data[nans] = 0 
        elif np.any(nans):
            data[nans] = np.interp(x(nans), x(~nans), data[~nans])

        past_target = rearrange(
            torch.as_tensor(data, dtype=torch.float32), "t -> 1 t 1"
        )
        # 1s if the value is observed, 0s otherwise. Shape: (batch, time, variate)
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
        past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)
        
        forecast = self.model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )
        forecast = forecast.mean(axis=[0,1]).numpy()

        return None, forecast, None

if __name__ == "__main__":
    print("Hw")
    model = MambaTSFM()
    pdb.set_trace()


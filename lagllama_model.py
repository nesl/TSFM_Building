import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from itertools import islice
import sys
import os

# Add the lag_src directory to the system path
sys.path.append(os.path.abspath("./lag_src/"))
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
# Now you can import the modules
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from lag_llama.gluon.estimator import LagLlamaEstimator

class LagLlamaModel:
    def __init__(self, checkpoint_path="./lag_src/lag-llama.ckpt"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda') 
        self.scaler = MinMaxScaler()

    def _prepare_data(self, data, item_id, sampling_rate, target_name='target', normalize=True):
        # Separate the values and timestamps
        values = np.array(data[:,0]).reshape(-1, 1)
        if normalize:
            normalized_data = self.scaler.fit_transform(values).flatten()
        else: 
            normalized_data = values.flatten()
        timestamps = data[:, -1]

        # Convert numpy array to pandas dataframe using the provided timestamps
        timestamps = pd.to_datetime(timestamps)
        df = pd.DataFrame(normalized_data, columns=[target_name], index=timestamps)
        df['timestamp'] = df.index
        df['item_id'] = item_id

        # Infer frequency from sampling rate (in seconds)
        freq = f'{sampling_rate}S'

        # Convert DataFrame to ListDataset
        dataset = ListDataset(
            [
                {
                    "start": df['timestamp'].iloc[0],
                    "target": df[target_name].values,
                    "item_id": item_id
                }
            ],
            freq=freq
        )
        return dataset

    def get_lag_llama_predictions(self, dataset, prediction_length, context_length, use_rope_scaling=False, num_samples=100):
        print('in get_lag_llama_predictions')
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)  # Uses GPU since in this Colab we use a GPU.
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        
        rope_scaling_arguments = {
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        }
        print('rope scaling arguments')
        estimator = LagLlamaEstimator(
            ckpt_path=self.checkpoint_path,
            prediction_length=prediction_length,
            context_length=context_length,  # Lag-Llama was trained with a context length of 32, but can work with any context length

            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            rope_scaling=rope_scaling_arguments if use_rope_scaling else None,

            batch_size=1,
            num_parallel_samples=100,
            device=self.device,
        )
        print('estimator created')
        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)
        print('predictor created')
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=predictor,
            num_samples=num_samples
        )
        print('forecasts and ts')
        forecasts = list(forecast_it)
        tss = list(ts_it)
        print('forecasts and ts returned')
        return forecasts, tss

    def __call__(self, data, item_id, sampling_rate, prediction_length, use_rope_scaling=False, num_samples=100, normalize=True):
        print(' in function call')
        dataset = self._prepare_data(data, item_id, sampling_rate, normalize=normalize)
        len_data=len(data)
        print('data prepared')
        forecasts, tss = self.get_lag_llama_predictions(
            dataset, 
            prediction_length, 
            context_length=len_data, 
            use_rope_scaling=use_rope_scaling, 
            num_samples=num_samples
        )
        print('got forecasts')
        if normalize:
            # Extract the forecasted normalized values
            normalized_forecast = np.array([forecast.mean for forecast in forecasts]).flatten()

            # Denormalize the forecasted values
            denormalized_forecast = self.scaler.inverse_transform(normalized_forecast.reshape(-1, 1)).flatten()
        else:
            denormalized_forecast = np.array([forecast.mean for forecast in forecasts]).flatten()
        print('denormalized')
        return denormalized_forecast

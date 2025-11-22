"""
Baseline Models for Indoor Temperature Forecasting with Covariates

This script trains all baseline models for indoor temperature prediction using
external covariates (outdoor temperature, solar irradiance, HVAC control signals).

Models trained:
  - R2C2: 2 Resistances, 2 Capacitances thermal model
  - Ridge Regression: Multivariate (with covariates) and Univariate versions
  - Random Forest: Multivariate (with covariates) and Univariate versions

Usage:
    python train_forecasting_baselines.py --noise 0.1 --pred_hrz 64 --duration 448

    # Run all noise levels
    python train_forecasting_baselines.py --all_noise
"""

import pandas as pd
import numpy as np
import os
import warnings
from collections import OrderedDict
import argparse

from scipy.linalg import expm, inv
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


# ============================================================================
# Model Definitions
# ============================================================================

class R2C2Model:
    """
    R2C2 (2 Resistances, 2 Capacitances) thermal model for building temperature prediction.
    """
    def __init__(self, Ri, Re, Ci, Ce, Ai, Ae, roxP_hvac):
        self.Ri = Ri
        self.Re = Re
        self.Ci = Ci
        self.Ce = Ce
        self.Ai = Ai
        self.Ae = Ae
        self.roxP_hvac = roxP_hvac
        self.N_states = 2
        self.update_matrices()

    def update_matrices(self):
        """Update continuous-time state-space matrices."""
        self.Ac = np.array([[-1/(self.Ci*self.Ri), 1/(self.Ci*self.Ri)],
                            [1/(self.Ce*self.Ri), -1/(self.Ce*self.Ri) - 1/(self.Ce*self.Re)]])
        self.Bc = np.array([[0, -self.roxP_hvac / self.Ci, self.Ai / self.Ci],
                            [1/(self.Ce*self.Re), 0, self.Ae/self.Ce]])
        self.Cc = np.array([[1, 0]])

    def discretize(self, dt):
        """Discretize continuous-time model."""
        n = self.N_states
        F = expm(self.Ac * dt)
        G = np.dot(inv(self.Ac), np.dot(F - np.eye(n), self.Bc))
        H = self.Cc
        return F, G

    def predict_onestep(self, T, Te, T_ext, u, ghi, dt):
        """One-step temperature prediction using vectorized operations."""
        F, G = self.discretize(dt)
        state_matrix = np.vstack((T, Te)).T
        input_matrix = np.vstack((T_ext, u, ghi)).T
        predictions = (F @ state_matrix.T) + (G @ input_matrix.T)
        predictions = predictions.T
        return predictions[:, 0]

    def predict_two(self, T, Te, T_ext, u, ghi, dt):
        """Predict both indoor and envelope temperatures."""
        F, G = self.discretize(dt)
        state_matrix = np.vstack((T, Te)).T
        input_matrix = np.vstack((T_ext, u, ghi)).T
        predictions = (F @ state_matrix.T) + (G @ input_matrix.T)
        predictions = predictions.T
        return predictions[:, 0], predictions[:, 1]

    def autoregressive_predict(self, T_init, Te_init, T_ext_seq, u_seq, ghi_seq, dt, pred_hrz=64):
        """Autoregressive temperature predictions."""
        T = np.zeros(pred_hrz)
        Te = Te_init
        preds = []

        for t in range(0, pred_hrz):
            if t == 0:
                T_pred = self.predict_onestep(
                    T_init, Te_init,
                    T_ext_seq[0], u_seq[0], ghi_seq[0], dt
                )
            else:
                T_pred, _ = self.predict_two(
                    T_prev, Te,
                    T_ext_seq[t], u_seq[t], ghi_seq[t], dt
                )
            preds.append(T_pred)
            T_prev = T_pred
            Te = (T_pred + T_ext_seq[t]) / 2

        # Convert Kelvin to Fahrenheit
        preds = (np.array(preds) - 273.15) * 9/5 + 32

        return preds


# ============================================================================
# Data Loading and Processing
# ============================================================================

def read_optimization_results_from_csv(sensor_counts):
    """Reads R2C2 optimization results from CSV files."""
    optimization_results = {}
    for sensor_count in sensor_counts:
        file_name = f"/Users/ozanbaris/Documents/GitHub/TS-foundation-model/R2C2_models/decent_R2C2_optimization_results_sensor_{sensor_count}.csv"
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            houses = {}
            for _, row in df.iterrows():
                house_id = row['House ID']
                results = {
                    'rmse_train': row['RMSE Train'],
                    'rmse_test': row['RMSE Test'],
                    'optimal_params': {
                        'Ri': list(map(float, row['Ri_values'].split(', '))),
                        'Re': list(map(float, row['Re_values'].split(', '))),
                        'Ci': list(map(float, row['Ci_values'].split(', '))),
                        'Ce': list(map(float, row['Ce_values'].split(', '))),
                        'Ai': list(map(float, row['Ai_values'].split(', '))),
                        'Ae': list(map(float, row['Ae_values'].split(', '))),
                        'roxP_hvac': list(map(float, row['roxP_hvac'].split(', ')))
                    }
                }
                houses[house_id] = results

            optimization_results[sensor_count] = houses
        else:
            print(f"File '{file_name}' not found.")

    return optimization_results


def read_csvs_to_dfs(main_output_directory):
    """Reads all house CSV files from subdirectories."""
    all_houses_dict = {}

    for subdirectory in os.listdir(main_output_directory):
        sub_output_directory = os.path.join(main_output_directory, subdirectory)

        if not os.path.isdir(sub_output_directory):
            continue

        house_group = int(subdirectory.split("_")[-1])

        if house_group not in all_houses_dict:
            all_houses_dict[house_group] = {}

        for filename in os.listdir(sub_output_directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(sub_output_directory, filename)
                house_id = filename.split("_")[-1].replace(".csv", "")
                df = pd.read_csv(file_path)
                all_houses_dict[house_group][house_id] = df

    return all_houses_dict


def process_house_data(df):
    """Processes a single house DataFrame."""
    df['duty_cycle'] = df['CoolingRunTime'] / 3600
    df.rename(columns={'Outdoor_Temperature': 'Text'}, inplace=True)

    sensor_rename_map = {
        'Thermostat_Temperature': 'T01_TEMP',
        'RemoteSensor1_Temperature': 'T02_TEMP',
        'RemoteSensor2_Temperature': 'T03_TEMP',
        'RemoteSensor3_Temperature': 'T04_TEMP',
        'RemoteSensor4_Temperature': 'T05_TEMP',
        'RemoteSensor5_Temperature': 'T06_TEMP',
    }
    df.rename(columns=sensor_rename_map, inplace=True)

    temp_columns = [f"T0{i}_TEMP" for i in range(1, 7)] + ['Text']
    for col in temp_columns:
        df[col] = (df[col] - 32) * 5/9 + 273.15

    columns_to_keep = ['time', 'GHI', 'duty_cycle'] + temp_columns
    df = df[columns_to_keep]
    df.fillna(method='ffill', inplace=True)

    return df


# ============================================================================
# Metrics and Results
# ============================================================================

def save_results_for_model(model_name, ground_truth, timestamp_forecast, forecast,
                           duration, pred_hrz, mode, occupancy, batch_id, directory):
    """Save model predictions and metrics to CSV file."""
    mse = np.mean((ground_truth - forecast) ** 2)
    rmse = np.sqrt(mse)

    result = {
        'batch_id': batch_id,
        'MSE': mse,
        'RMSE': rmse,
        'GroundTruth': ground_truth.tolist(),
        'Timestamp_forecast': timestamp_forecast.tolist(),
        'Forecast': forecast.tolist(),
        'Duration': duration,
        'PredictionHorizon': pred_hrz,
        'Mode': mode,
        'Occupancy': occupancy,
        'Model': model_name
    }

    file_path = f"{directory}/results_{mode}_{occupancy}/{duration}_{pred_hrz}/{model_name}.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['batch_id', 'MSE', 'RMSE', 'Mode',
                                   'Occupancy', 'Duration', 'PredictionHorizon',
                                   'GroundTruth', 'Timestamp_forecast',
                                   'Forecast', 'Model'])

    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    df.to_csv(file_path, index=False)


# ============================================================================
# Autoregressive Prediction Functions
# ============================================================================

def autoregressive_predict_ridge(ridge_model, test_df, pred_hrz=24):
    """Autoregressive forecast for Ridge regression with covariates."""
    preds_K = []
    T_init = test_df['T01_TEMP'].iloc[0]
    T_prev = T_init

    for t in range(0, pred_hrz):
        features = {
            'Text': test_df['Text'].iloc[t],
            'duty_cycle': test_df['duty_cycle'].iloc[t],
            'GHI': test_df['GHI'].iloc[t],
            'T01_TEMP': T_prev
        }
        X = pd.DataFrame([features])
        T_next = ridge_model.predict(X)[0]
        preds_K.append(T_next)
        T_prev = T_next

    T_pred_F = (np.array(preds_K) - 273.15) * 9/5 + 32
    return T_pred_F


def autoregressive_predict_rf(rf_model, test_df, pred_hrz=24):
    """Autoregressive forecast for Random Forest with covariates."""
    preds_K = []
    T_init = test_df['T01_TEMP'].iloc[0]
    T_prev = T_init

    for t in range(0, pred_hrz):
        features = {
            'Text': test_df['Text'].iloc[t],
            'duty_cycle': test_df['duty_cycle'].iloc[t],
            'GHI': test_df['GHI'].iloc[t],
            'T01_TEMP': T_prev
        }
        X = pd.DataFrame([features])
        T_next = rf_model.predict(X)[0]
        preds_K.append(T_next)
        T_prev = T_next

    T_pred_F = (np.array(preds_K) - 273.15) * 9/5 + 32
    return T_pred_F


def autoregressive_predict_ridge_univariate(ridge_model, test_df, pred_hrz=24):
    """Univariate autoregressive forecast for Ridge."""
    preds_K = []
    T_prev = test_df['T01_TEMP'].iloc[0]

    for _ in range(pred_hrz):
        X = pd.DataFrame({'T01_TEMP': [T_prev]})
        T_next = ridge_model.predict(X)[0]
        preds_K.append(T_next)
        T_prev = T_next

    T_pred_F = (np.array(preds_K) - 273.15) * 9/5 + 32
    return T_pred_F


def autoregressive_predict_rf_univariate(rf_model, test_df, pred_hrz=24):
    """Univariate autoregressive forecast for Random Forest."""
    preds_K = []
    T_prev = test_df['T01_TEMP'].iloc[0]

    for _ in range(pred_hrz):
        X = pd.DataFrame({'T01_TEMP': [T_prev]})
        T_next = rf_model.predict(X)[0]
        preds_K.append(T_next)
        T_prev = T_next

    T_pred_F = (np.array(preds_K) - 273.15) * 9/5 + 32
    return T_pred_F


# ============================================================================
# Training and Testing Pipeline
# ============================================================================

def training_and_testing_baselines(all_houses_reduced, optimization_results_R2C2_decent,
                                   processed_houses_reduced, pred_hrz, duration, noise,
                                   mode, batch_id):
    """Train and test R2C2, Ridge, and Random Forest models."""
    # Flatten and sort the house data
    new_house_data = {}
    total = 0
    house_data = OrderedDict(sorted(all_houses_reduced.items()))

    for house_group in house_data:
        sorted_dict = OrderedDict(sorted(house_data[house_group].items()))
        for house_id in sorted_dict:
            total += 1
            new_house_data[house_id] = total

    # Reorganize the dictionary to remove the sensor_count key
    flattened_results = {}
    for sensor_count, houses in optimization_results_R2C2_decent.items():
        for house_id, result in houses.items():
            flattened_results[house_id] = result

    flattened_data = {}
    for sensor_count, houses in processed_houses_reduced.items():
        for house_id, df in houses.items():
            flattened_data[house_id] = df

    # Process each house
    for house_id, house_data_df in flattened_data.items():
        rank = new_house_data[house_id]
        print(f"\nHouse ID {house_id} with rank {rank}")

        if house_id not in flattened_results:
            print(f"Skipping house {house_id} (no R2C2 results).")
            continue

        # Extract R2C2 parameters and initialize model
        results = flattened_results[house_id]
        optimal_params = results['optimal_params']

        Ri = optimal_params['Ri'][0]
        Re = optimal_params['Re'][0]
        Ci = optimal_params['Ci'][0]
        Ce = optimal_params['Ce'][0]
        Ai = optimal_params['Ai'][0]
        Ae = optimal_params['Ae'][0]
        roxP_hvac = optimal_params['roxP_hvac'][0]

        r2c2_model = R2C2Model(Ri, Re, Ci, Ce, Ai, Ae, roxP_hvac)

        # Train/test split
        test_size = 0.125
        num_test_samples = int(len(house_data_df) * test_size)
        split_index = len(house_data_df) - num_test_samples

        train_df = house_data_df.iloc[224:split_index].copy()
        test_df = house_data_df.iloc[split_index:].copy()
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # R2C2 Autoregressive predictions
        T_init = test_df['T01_TEMP'].iloc[0]
        T_ext_init = test_df['Text'].iloc[0]
        Te_init = (T_init + T_ext_init) / 2

        T_ext_seq = test_df['Text'].iloc[:pred_hrz].values
        u_seq = test_df['duty_cycle'].iloc[:pred_hrz].values
        ghi_seq = test_df['GHI'].iloc[:pred_hrz].values

        predictions_r2c2 = r2c2_model.autoregressive_predict(
            T_init, Te_init, T_ext_seq, u_seq, ghi_seq, dt=3600, pred_hrz=pred_hrz
        )

        # Ground truth in Fahrenheit
        ground_truth_K = test_df['T01_TEMP'].iloc[1:pred_hrz+1].values
        ground_truth_F = (ground_truth_K - 273.15) * 9/5 + 32

        rmse_r2c2 = np.sqrt(mean_squared_error(ground_truth_F, predictions_r2c2))
        print(f"[R2C2] RMSE = {rmse_r2c2:.2f} F")

        model_name = "R2C2"
        directory = f'/Users/ozanbaris/Documents/GitHub/TS-foundation-model/Aug18_ecobee_results_{noise}/{model_name}_new_ecobee_cov_{noise}_csv'
        save_results_for_model(
            model_name=model_name,
            ground_truth=ground_truth_F,
            timestamp_forecast=test_df.index[:pred_hrz],
            forecast=predictions_r2c2,
            duration=duration,
            pred_hrz=pred_hrz,
            mode=mode,
            occupancy=rank,
            batch_id=batch_id,
            directory=directory
        )

        # Prepare training data for Ridge & RF (multivariate)
        X_train = train_df[['Text', 'duty_cycle', 'GHI', 'T01_TEMP']][:-1]
        y_train = train_df['T01_TEMP'][1:]
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # Fit and test Ridge (multivariate)
        ridge_model = Ridge()
        ridge_model.fit(X_train, y_train)

        predictions_ridge = autoregressive_predict_ridge(
            ridge_model, test_df, pred_hrz=pred_hrz
        )
        rmse_ridge = np.sqrt(mean_squared_error(ground_truth_F, predictions_ridge))
        print(f"[Ridge] RMSE = {rmse_ridge:.2f} F")

        model_name = "Ridge"
        directory = f'/Users/ozanbaris/Documents/GitHub/TS-foundation-model/Aug18_ecobee_results_{noise}/{model_name}_new_ecobee_cov_{noise}_csv'
        save_results_for_model(
            model_name=model_name,
            ground_truth=ground_truth_F,
            timestamp_forecast=test_df.index[:pred_hrz],
            forecast=predictions_ridge,
            duration=duration,
            pred_hrz=pred_hrz,
            mode=mode,
            occupancy=rank,
            batch_id=batch_id,
            directory=directory
        )

        # Fit and test Random Forest (multivariate)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        predictions_rf = autoregressive_predict_rf(
            rf_model, test_df, pred_hrz=pred_hrz
        )
        rmse_rf = np.sqrt(mean_squared_error(ground_truth_F, predictions_rf))
        print(f"[RF] RMSE = {rmse_rf:.2f} F")

        model_name = "RandomForest"
        directory = f'/Users/ozanbaris/Documents/GitHub/TS-foundation-model/Aug18_ecobee_results_{noise}/{model_name}_new_ecobee_cov_{noise}_csv'
        save_results_for_model(
            model_name=model_name,
            ground_truth=ground_truth_F,
            timestamp_forecast=test_df.index[:pred_hrz],
            forecast=predictions_rf,
            duration=duration,
            pred_hrz=pred_hrz,
            mode=mode,
            occupancy=rank,
            batch_id=batch_id,
            directory=directory
        )

        # Prepare univariate training data
        X_train_univ = train_df[['T01_TEMP']][:-1]
        print(f"X_train (Univariate) shape: {X_train_univ.shape}, y_train shape: {y_train.shape}")

        # Fit and test Ridge (univariate)
        ridge_model_univ = Ridge()
        ridge_model_univ.fit(X_train_univ, y_train)

        predictions_ridge_univ = autoregressive_predict_ridge_univariate(
            ridge_model_univ, test_df, pred_hrz=pred_hrz
        )
        rmse_ridge_univ = np.sqrt(mean_squared_error(ground_truth_F, predictions_ridge_univ))
        print(f"[Ridge_univ] RMSE = {rmse_ridge_univ:.2f} F")

        model_name = "Ridge_univ"
        directory = f'/Users/ozanbaris/Documents/GitHub/TS-foundation-model/Aug18_ecobee_results_{noise}/{model_name}_new_ecobee_cov_{noise}_csv'
        save_results_for_model(
            model_name=model_name,
            ground_truth=ground_truth_F,
            timestamp_forecast=test_df.index[:pred_hrz],
            forecast=predictions_ridge_univ,
            duration=duration,
            pred_hrz=pred_hrz,
            mode=mode,
            occupancy=rank,
            batch_id=batch_id,
            directory=directory
        )

        # Fit and test Random Forest (univariate)
        rf_model_univ = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model_univ.fit(X_train_univ, y_train)

        predictions_rf_univ = autoregressive_predict_rf_univariate(
            rf_model_univ, test_df, pred_hrz=pred_hrz
        )
        rmse_rf_univ = np.sqrt(mean_squared_error(ground_truth_F, predictions_rf_univ))
        print(f"[RF_univ] RMSE = {rmse_rf_univ:.2f} F")

        model_name = "RF_univ"
        directory = f'/Users/ozanbaris/Documents/GitHub/TS-foundation-model/Aug18_ecobee_results_{noise}/{model_name}_new_ecobee_cov_{noise}_csv'
        save_results_for_model(
            model_name=model_name,
            ground_truth=ground_truth_F,
            timestamp_forecast=test_df.index[:pred_hrz],
            forecast=predictions_rf_univ,
            duration=duration,
            pred_hrz=pred_hrz,
            mode=mode,
            occupancy=rank,
            batch_id=batch_id,
            directory=directory
        )

        print(f"Finished House {rank}\n{'-'*40}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train R2C2 baseline models')
    parser.add_argument('--noise', type=float, default=0, help='Noise level (default: 0)')
    parser.add_argument('--all_noise', action='store_true', help='Run all noise levels')
    parser.add_argument('--pred_hrz', type=int, default=64, help='Prediction horizon (default: 64)')
    parser.add_argument('--duration', type=int, default=448, help='Duration (default: 448)')
    parser.add_argument('--batch_id', type=int, default=0, help='Batch ID (default: 0)')

    args = parser.parse_args()

    # Load R2C2 optimization results
    print("Loading R2C2 optimization results...")
    sensor_counts = [1, 2, 3, 4, 5]
    optimization_results_R2C2_decent = read_optimization_results_from_csv(sensor_counts)

    # Determine which noise levels to process
    if args.all_noise:
        noise_levels = [0, 0.1, 0.2, 0.5, 1, 2, 5]
    else:
        noise_levels = [args.noise]

    mode = None

    for noise in noise_levels:
        print(f"\n{'='*60}")
        print(f"Processing Noise Level: {noise}")
        print(f"{'='*60}\n")

        dataset_name = f'house_data_csvs_{noise}'
        main_output_directory = f"/Users/ozanbaris/Documents/GitHub/TS-foundation-model/{dataset_name}"

        all_houses_reduced = read_csvs_to_dfs(main_output_directory)

        processed_houses_reduced = {}
        for sensor_count, houses in all_houses_reduced.items():
            processed_houses = {}
            for house_id, df in houses.items():
                if 'GHI' not in df.columns:
                    print(f"Skipping house {house_id} due to missing 'GHI' column.")
                    continue
                processed_houses[house_id] = process_house_data(df.copy())
            processed_houses_reduced[sensor_count] = processed_houses

        training_and_testing_baselines(
            all_houses_reduced,
            optimization_results_R2C2_decent,
            processed_houses_reduced,
            args.pred_hrz,
            args.duration,
            noise,
            mode,
            args.batch_id
        )

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)

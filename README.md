# TSFM_Building

## 🔥 News

<!-- - 🔥 **v2.0** - Extension: \[Data-Centric Engineering Journal\] Can Time-Series Foundation Models Perform Building Energy Management Tasks? -->
- 🔥 **v2.0** [11/20/26] - We have reorganized the repository to incorporate the extended journal version, *Can Time-Series Foundation Models Perform Building Energy Management Tasks?* Our extension preprint is available on [ArXiv](https://arxiv.org/abs/2506.11250). Stay tuned for the final published version.
- **v1.0** - BuildSys 2024 release: \[BuildSys '24\] Are Time Series Foundation Models Ready to Revolutionize Predictive Building Analytics? 

### 🔥 We included classification evaluation using TSFM in **v2.0**. 
### Classification 

```bash
# Create the environment from the YAML file
cd ./classification
conda env create -f nesl_environment.yml

# Activate it
conda activate nesl
```

- Follow the instruction in [soft DTW](https://github.com/Maghoumi/pytorch-softdtw-cuda) to install the package.
- Download the [WHITED dataset](https://www.cs.cit.tum.de/dis/resources/whited/) (WHITEDv1.1.zip)
- Copy and unzip the dataset:
```bash
mv WHITEDv1.1.zip ./classification/data/
cd ./classification
unzip ./data/WHITEDv1.1.zip
```
- Perform training using baselines `resnet`, `dtw`, and `ts2vec`:
```bash
python train_ts2vec_whited.py
python train_dtw_whited.py
python train_resnet_whited.py
```
- Perform evaluation using TSFM `chronos` and `moment`:
```bash
# MOMENT model - generates embeddings + trains SVM classifier
python train_tsfm_whited.py --model moment

# Chronos model - generates embeddings + trains SVM classifier
python train_tsfm_whited.py --model chronos

# To save embeddings to .npy files, add --save_embeddings flag:
python train_tsfm_whited.py --model moment --save_embeddings
# This saves: moment_train_embeddings_whited.npy, moment_test_embeddings_whited.npy, and labels

python train_tsfm_whited.py --model chronos --save_embeddings
# This saves: chronos_train_embeddings_whited.npy, chronos_test_embeddings_whited.npy, and labels
```

- Perform evaluation using TSFM `chronos` and `moment` on BTS dataset:
```bash
# Note: We used the BTS dataset for our second classification task.
# At the time of the experiment, all of the data was not public due to an ongoing competition.
# Thus we used the data from the competition: https://www.aicrowd.com/challenges/brick-by-brick-2024
# Download train_X_v0.1.0.zip and train_y_v0.1.0.csv from the competition
# Place them in the ./classification/data/ folder

# MOMENT model - generates embeddings + trains SVM classifiers (multi-label)
python train_tsfm_bts.py --model moment

# Chronos model - generates embeddings + trains SVM classifiers (multi-label)
python train_tsfm_bts.py --model chronos

# To save embeddings to .npy files, add --save_embeddings flag:
python train_tsfm_bts.py --model moment --save_embeddings
# This saves: moment_train_embeddings_bts.npy, moment_test_embeddings_bts.npy, and labels

python train_tsfm_bts.py --model chronos --save_embeddings
# This saves: chronos_train_embeddings_bts.npy, chronos_test_embeddings_bts.npy, and labels
```

### Forecasting with Covariates

For the forecasting with covariates task, we use a sampled version of the Ecobee dataset (linked below). The original R2C2 implementation is available at [this codebase](https://github.com/ozanbarism/GBM4SingleZoneMultiNodeSystems). To prepare the dataset, clone the R2C2 repository and run the sampling script:

```bash
# Clone the R2C2 repository
git clone https://github.com/ozanbarism/GBM4SingleZoneMultiNodeSystems.git
cd GBM4SingleZoneMultiNodeSystems

# Run the sampling script (requires the Ecobee dataset downloaded first)
python sample_ecobee_for_covariate.py
```

This will generate a folder called `house_data_csvs` with different noise levels (e.g., `house_data_csvs_0`, `house_data_csvs_0.1`, etc.) that models for forecasting with covariates will use. 

We evaluate the forecasting performance of three TSFMs: Uni2TS, TimesFM, and TimeGPT, which are the only models in this study that explicitly support covariate-based predictions. For TimesFM, we use the Dec 30, 2024 release shown [here](https://github.com/google-research/timesfm/tree/1e249ef0b167a309d4f2ff37efd9d433d34c31dc?tab=readme-ov-file#update---dec-30-2024) with the Xreg + TSFM approach. Uni2TS (Moirai) and TimeGPT inherently support covariates so we use them as is.

```bash
# Create the environment from the YAML file
conda env create -f covariate_environment.yml

# Activate it
conda activate covariate_env
```

- Perform training using baseline forecasting models with covariates (R2C2, Ridge, Random Forest):
```bash
# Note: First prepare the dataset using sample_ecobee_for_covariate.py as described above

# Train all baseline models on a specific noise level
python train_forecasting_baselines.py --noise 0.1 --pred_hrz 64 --duration 448

# Train on all noise levels (0, 0.1, 0.2, 0.5, 1, 2, 5)
python train_forecasting_baselines.py --all_noise
```

# Usage
The following code will run inference on each model and dataset:
```bash
python main.py --model YOUR_MODEL --real_data DATA
```
Here,
```
real_data \in {ecobee, electricity_uci, umass}
```
```
model \in {AutoARIMA,
              SeasonaARIMA,
              moment,
              chronos,
              TimeGPT,
              TimesFM,
              uni2ts
              ...}
```


# Download dataset
## Download electricity_uci dataset
Download the dataset from [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
Then put ```LD2011_2014.txt``` in the ```./data/``` folder.

## Download ecobee dataset
Download the dataset [Ecobee](https://www.osti.gov/biblio/1854924) and extract it. Then run the sampling script for univariate forecasting:

```bash
# Run the sampling script (from the root directory of this repository)
python sample_ecobee_for_univariate.py
```

This will generate ```combined_thermostat_data.csv```. Move it to the ```./data/``` folder:

```bash
mv combined_thermostat_data.csv ./data/
``` 

## Download smart* dataset
The [smart* (umass) dataset](https://traces.cs.umass.edu/docs/traces/smartstar/) has been uploaded with the repo. We downloaded the apartment dataset called ' apartment-electrical.tar.gz' and merged the data for years 2015 and 2016. It uses CC BY 4.0 license, which allows copy and redistribute the material.

# TS-foundation-model Setup
To reproduce the results, please first setup environments for **each** model:
## Chronos installation
 You can find it in [this repo](https://github.com/amazon-science/chronos-forecasting?tab=readme-ov-file)

Create an environment for Chronos
```bash
virtualenv chronos -p python3.10
source chronos/bin/activate
```
Install general packages
```bash
pip install -r requirements.txt
```
Install chronos
```bash
python3.10 -m pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

 ## Moment installation
 You can find it in [this repo](https://github.com/moment-timeseries-foundation-model/moment.git)

Create a environment for Moment. Please note that only python version >= 3.10 is supported.
```bash
virtualenv moment -p python3.10
source moment/bin/activate
```
Install general packages
```bash
pip install -r requirements.txt
```
Install moment
```bash
python3.10 -m pip install git+https://github.com/moment-timeseries-foundation-model/moment.git
```

## Uni2ts installation

1. Clone repository:
```shell
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts
```

2) Create virtual environment:
```shell
virtualenv uni2ts -p python3.10
source uni2ts/bin/activate
```

3) Build from source:
```shell
pip install -e '.[notebook]'
```

Install general packages
```bash
pip install -r requirements.txt
```

 ## TimeGPT installation

 Install general packages
```bash
pip install -r requirements.txt
```

```bash
pip install nixtla>=0.5.1
```
need to get api key from nixtla
```python
import pandas as pd
from nixtla import NixtlaClient

# 1. Instantiate the NixtlaClient
nixtla_client = NixtlaClient(api_key = 'YOUR API KEY HERE')
```
 ## TimesFM installation

View the original [repo](https://github.com/google-research/timesfm?tab=readme-ov-file). Follow the instruction in the repo to install the model.

<!-- Create an enviroment using the yaml file.
```
conda env create --file=tfm_environment.yml
conda activate TimesFM
```

Install general packages
```
pip install -r requirements.txt
```

Install TimesFM. You maye need to sure you have installed pytorch-cuda==12.1.
```
python3.10 -m pip install git+https://github.com/google-research/timesfm.git
``` -->
\[Update on 09/30/24\] TimesFM can be installed via ```pip install timesfm```

## LagLlama installation
Create a new conda env
```bash
# Alternative with conda:
# conda create -n LagLlama python=3.10
# conda activate LagLlama

# Using virtualenv:
virtualenv LagLlama -p python3.10
source LagLlama/bin/activate
```

Install general packages
```bash
pip install -r requirements.txt
```

clone the repo
```bash
git clone https://github.com/time-series-foundation-models/lag-llama/
```
Go to the current folder
```bash
cd <Current_folder>
```
Hyperlink your lag-llama folder to the current folder
```bash
ln -s <Your_lag-llama_folder> lag_src
```
Copy the model file and go to the lag-llama folder
```bash
cp lagllama_model.py lag_src
cd ./lag_src
```
install the requirements
```bash
pip install -r requirements.txt --quiet
# This could take some time
```
download pretrained model weights from HuggingFace 🤗
```bash
huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir ./
```
and it will copy the model file to lag-llama

## Contact Information

If you have any questions or feedback, feel free to reach out:

- **Name:** Ozan Baris Mulayim
- **Email:** [omulayim@andrew.cmu.edu](mailto:omulayim@andrew.cmu.edu)

- **Name:** Pengrui Quan
- **Email:** [prquan@ucla.edu](mailto:prquan@ucla.edu)

## License

This dataset is released under the BSD 3-Clause License. See the LICENSE file for details.

## Acknowledgement
Mario Bergés and Mani Srivastava hold concurrent appointments as Amazon Scholars, and as Professors at their respective universities, but work in this paper is not associated with Amazon. Dezhi Hong is also affiliated with Amazon but work in this paper is not associated with Amazon. This research was sponsored in part by AFOSR award \#FA95502210193, DEVCOM ARL award \#W911NF1720196, NSF award \#CNS-2325956, NIH award \#P41EB028242, and Sandia National Laboratories award \#2169310.

We would like to extend our thanks to authors of [ts2vec](https://github.com/zhihanyue/ts2vec), from which the baseline is built on.

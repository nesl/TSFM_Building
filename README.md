# TSFM_Building

Pytorch implementation of \[BuildSyS '24\] Are Time Series Foundation Models Ready to Revolutionize Predictive Building Analytics? 

# Usage
```
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
              ...}
```

# Download dataset
## Download ElectricityLoadDiagrams20112014
Download the dataset from [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
Then put ```LD2011_2014.txt``` in the main folder.

## Download Ecobee dataset
Download the dataset from [Ecobee](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
Then put ```combined_thermostat_data.csv``` in the main folder.

# TS-foundation-model Setup
## Chronos installation
 You can find it in [this repo](https://github.com/amazon-science/chronos-forecasting?tab=readme-ov-file)

Create an environment for Chronos
```
conda create -n chronos python=3.10
conda activate chronos
```
Install general packages
```
pip install -r requirements.txt
```
Install chronos
 ```
python3.10 -m pip install git+https://github.com/amazon-science/chronos-forecasting.git
 ```

 ## Moment installation
 You can find it in [this repo](pip install git+https://github.com/moment-timeseries-foundation-model/moment.git)

Create a environment for Moment. Please note that only python version >= 3.10 is supported.
```
conda create -n moment python=3.10
conda activate moment
```
Install general packages
```
pip install -r requirements.txt
```
Install moment
 ```
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
conda create -n uni2ts python=3.10
conda activate uni2ts
```

3) Build from source:
```shell
pip install -e '.[notebook]'
```

Install general packages
```
pip install -r requirements.txt
```

 ## TimeGPT installation

 Install general packages
```
pip install -r requirements.txt
```
 
 ```
pip install nixtla>=0.5.1
 ```
need to get api key from nixtla
 ```
import pandas as pd
from nixtla import NixtlaClient


# 1. Instantiate the NixtlaClient
nixtla_client = NixtlaClient(api_key = 'YOUR API KEY HERE')
 ```
 ## TimesFM installation

To view the original [repo](https://github.com/google-research/timesfm?tab=readme-ov-file)

Create an enviroment using the yaml file.
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
```
## LagLlama installation
Create a new conda env
```
conda create -n LagLlama python=3.10
conda activate LagLlama
```

Install general packages
```
pip install -r requirements.txt
```

clone the repo
```
!git clone https://github.com/time-series-foundation-models/lag-llama/
```
Hyperlink your lag-llama folder to the current folder
```
ln -s <Your_lag-llama_folder> ./lag_src
```
Copy the model file and go to the lag-llama folder
```
cp lagllama_model.py ./lag_src
cd ./lag_src
```
install the requirements
```
!pip install -r requirements.txt --quiet # this could take some time # ignore the errors displayed by colab
```
download pretrained model weights from HuggingFace 🤗
```
!huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir ./
```
copy the model file to lag-llama
## AutoARIMA & SeasonalARIMA
```
pip install pmdarima
```

## Install environment
```
pip install scipy
pip install statsmodels
```

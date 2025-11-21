# TSFM_Building

## 🔥 News

<!-- - 🔥 **v2.0** - Extension: \[Data-Centric Engineering Journal\] Can Time-Series Foundation Models Perform Building Energy Management Tasks? -->
- 🔥 **v2.0** [11/20/26] - We have reorganized the repository to incorporate the extended journal version, *Can Time-Series Foundation Models Perform Building Energy Management Tasks?* Our extension preprint is available on [ArXiv](https://arxiv.org/abs/2506.11250). Stay tuned for the final published version.
- **v1.0** - BuildSys 2024 release: \[BuildSys '24\] Are Time Series Foundation Models Ready to Revolutionize Predictive Building Analytics? 

### 🔥 We included classification evaluation using TSFM in **v2.0**. 
- Follow the instruction in [soft DTW](https://github.com/Maghoumi/pytorch-softdtw-cuda) to install the package.
- Download the WHITEDv1.1 dataset using the link.
- Copy and unzip the [whited dataset](https://www.cs.cit.tum.de/dis/resources/whited/)
```
mv WHITEDv1.1.zip ./classification/data/
cd ./classification
unzip ./data/WHITEDv1.1.zip
```
- Perform training using baselines `resent`, `dtw`, and `ts2vec`:
```
python train_ts2vec_whited.py
python train_dtw_whited.py
python train_resnet_whited.py
```
- Perform evaluation using TSFM `chronos` and `moment`:
```
python train_tsfm_whited.py --model chronos
```
or
```
python train_tsfm_whited.py --model moment
```
Note: We used the BTS dataset for our second classification task. At the time of the experiment, all of the data was not public due to an ongoing comptetiton. Thus we used the data ('data/train_X_v0.1.0.zip') from the [competition](https://www.aicrowd.com/challenges/brick-by-brick-2024#starter-kit-and-resources)

**📓 Jupyter Notebooks:**
- For TSFM embedding generation (MOMENT & Chronos) on BTS/WHITED datasets, see: `classification/Moment_embedding_generation.ipynb`
  - This notebook demonstrates how to generate MOMENT and Chronos embeddings from both datasets
  - For classification evaluation (SVM training, metrics), use the Python scripts:
    - `python train_tsfm_whited.py --model moment` (MOMENT embeddings + SVM classifier)
    - `python train_tsfm_whited.py --model chronos` (Chronos embeddings + SVM classifier)
- For R2C2 model experiments with covariates, see: `classification/covariates_R2C2.ipynb`

# Usage
The following code will run inference on each model and dataset:
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
              uni2ts
              ...}
```

Note: For R2C2 baseline experiments, see `classification/covariates_R2C2.ipynb`. The original R2C2 implementation is available at [this codebase](https://github.com/ozanbarism/GBM4SingleZoneMultiNodeSystems). 
# Download dataset
## Download electricity_uci dataset
Download the dataset from [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
Then put ```LD2011_2014.txt``` in the ```./data/``` folder.

## Download ecobee dataset
Download the dataset [Ecobee](https://drive.google.com/file/d/1nyfKfovXEHx1b_RH7Y8vT5yQUBxorYbn/view?usp=drive_link)
Then put ```combined_thermostat_data.csv``` in the ```./data/``` folder.

## Download smart* dataset
The smart* (umass) dataset has been uploaded with the repo.

# TS-foundation-model Setup
To reproduce the results, please first setup environments for **each** model:
## Chronos installation
 You can find it in [this repo](https://github.com/amazon-science/chronos-forecasting?tab=readme-ov-file)

Create an environment for Chronos
```
virtualenv chronos -p python3.10
source chronos/bin/activate
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
 You can find it in [this repo](https://github.com/moment-timeseries-foundation-model/moment.git)

Create a environment for Moment. Please note that only python version >= 3.10 is supported.
```
virtualenv moment -p python3.10
source moment/bin/activate
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
virtualenv uni2ts -p python3.10
source uni2ts/bin/activate
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
```
<!-- conda create -n LagLlama python=3.10
conda activate LagLlama -->
virtualenv LagLlama -p python3.10
source LagLlama/bin/activate
```

Install general packages
```
pip install -r requirements.txt
```

clone the repo
```
!git clone https://github.com/time-series-foundation-models/lag-llama/
```
Go to the current folder
```
cd <Current_folder>
```
Hyperlink your lag-llama folder to the current folder
```
ln -s <Your_lag-llama_folder> lag_src
```
Copy the model file and go to the lag-llama folder
```
cp lagllama_model.py lag_src
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

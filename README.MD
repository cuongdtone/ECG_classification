# ECG Classification
12 Leads ECG classification with LSTM
## Dev
```cuong_tran```
## Install
```buildoutcfg
conda create --name my_env python=3.8
conda activate my_env
pip install -r requirements.txt
```
## Usage
Datasets: 
- Train:
```buildoutcfg
python train.py --data-path PathToDatasets
```
- Test
```buildoutcfg
python evaluate.py --test-path
```
## Method
- Use basic LSTM module to classify single-lead 
- Comming soon: classification with multi-leads
## Evaluate


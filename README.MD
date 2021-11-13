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

```buildoutcfg
python train.py --data-path PathToDatasets
python evaluate.py --test-path
```
## Method
- Use basic LSTM module to classify single-lead 
- Comming soon: classification with multi-leads
## Evaluate
Single lead ECG:  
Train set:  
<img src="https://user-images.githubusercontent.com/79197775/141642710-8880c40e-7227-496e-82f0-d49648493985.png" alt="..." width="600" />  
Val set:  
<img src="https://user-images.githubusercontent.com/79197775/141642481-39ce0f47-9dab-4959-9519-2f5a4d49d901.png" alt="..." width="600" />  
Test set:  
<img src="https://user-images.githubusercontent.com/79197775/141642771-866f84e5-4e00-4d19-95be-4a196c31ce7a.png" alt="..." width="600" />

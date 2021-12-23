# Traning
Thank for author of https://github.com/onlyzdd/ecg-diagnosis  
Model at: models/resnet34_CPSC_all_42.pth   
Result at: result
## Evaluate:

### Requirements

- Python 3.7.4
- Matplotlib 3.1.1
- Numpy 1.17.2
- Pandas 0.25.2
- PyTorch 1.2.0
- Scikit-learn 0.21.3
- Scipy 1.3.1
- Shap 0.35.1
- Tqdm 4.36.1
- Wfdb 2.2.1

### Preprocessing

```sh
$ python preprocess.py --data-dir data/CPSC
```
### Deep model

```sh
$ python main.py --data-dir data/CPSC --leads all --use-gpu # training
$ python predict.py --data-dir data/CPSC --leads all --use-gpu # evaluation
```

### Interpretation

```sh
$ python shap_values.py --data-dir data/CPSC --use-gpu # visualizing shap values
```

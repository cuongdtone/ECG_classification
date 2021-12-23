from predict import Classifier
from glob import glob
import torch
from train import dataset
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier('last.ckpt', device)

#Path to data
test_list = glob('/home/cuong/Desktop/ECG/datasets_processing/*/*.mat')
test_data = dataset(test_list)
test_loader = DataLoader(test_data)
print(len(test_loader))

truth = []
pred = []
for i, (lead, label) in enumerate(test_loader):
    # print(lead)
    lead = lead.to(device)
    y = model.predict_evaluate(lead).tolist()
    label = label.tolist()
    truth.append(label[0])
    pred.append(y[0])

cm = metrics.confusion_matrix(truth, pred)
accuracy = metrics.accuracy_score(truth, pred)

df_cm = pd.DataFrame(cm, index = [i for i in "123456789"],
                  columns = [i for i in "123456789"])
plt.figure(figsize = (15,12))
sn.heatmap(df_cm, annot=True)
plt.title('Accuracy: %.2f %s '%(accuracy*100, '%'))
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.savefig('Confusion Matrix.png')
plt.show()



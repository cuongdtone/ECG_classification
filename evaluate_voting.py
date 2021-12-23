from predict import Classifier
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from plot import plot_cm, scores
import scipy.io as io


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier('last4.ckpt', device)

#Path to data
test_list = glob('/home/cuong/Desktop/ECG/datasets_split/test/*/*.mat')
print('Number of test data: ', len(test_list))

truth = []
pred = []
for i in test_list:
    # print(lead)
    lead = io.loadmat(i)
    y = model.predict_2(lead)
    label = i.split('/')[-2]
    label = int(label) - 1
    truth.append(label)
    pred.append(y[0])

#print(truth)
cm = metrics.confusion_matrix(truth, pred)
F1, F1x = scores(cm)

print('F1 = ', F1, end='\n')
for i in range(9):
    print('F1%d = '%(i+1), F1x[i])
accuracy = metrics.accuracy_score(truth, pred)

plot_cm(cm, accuracy=accuracy, normalize=False)
plt.show()


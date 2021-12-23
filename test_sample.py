import torch
from model import RNN
import torch.nn.functional as F
from scipy import io as io
from predict import Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Classifier('last.ckpt')

mat = io.loadmat('/home/cuong/Desktop/ECG/datasets/2/A0007.mat')

pred = model.predict(mat)
#print(pred)


def check_more():
    input_path = '/home/cuong/Desktop/ECG/datasets_split/test/9/'
    from glob import glob
    list_mat = glob(input_path + '/*.mat')
    c = 0
    for i in list_mat:
        mat = io.loadmat(i)
        pred = model.predict(mat)

        label = i.split('/')[-2]
        label = int(label) - 1
        #print(label, pred[0])
        if label == pred[0]:
            c+=1
    print(c)

check_more()


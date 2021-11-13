import torch
from model import RNN
import torch.nn.functional as F
from scipy import io as io
from predict import Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Classifier('model.ckpt')

mat = io.loadmat('datasets_test/test/1/A0229.mat')
pred = model.predict(mat)
print(pred)


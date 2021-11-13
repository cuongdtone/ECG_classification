import torch
from model import RNN
import torch.nn.functional as F
from scipy import io as io

class Classifier():
    def __init__(self, path, device = 'cpu'):
        self.len_leads = 4000
        self.device = device
        num_classes = 2
        self.model = RNN(self.len_leads, num_classes)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
    def predict(self, mat):
        mat = mat['ECG'][0][0]
        sex = mat[0][0]
        age = mat[1][0][0]
        data = mat[2][0]
        data = data[:self.len_leads]
        x = torch.from_numpy(data)
        x = torch.unsqueeze(x, dim=0)
        x = torch.unsqueeze(x, dim=0)
        x = x.to(self.device)

        y = self.model(x.float())
        y = F.softmax(y)
        return y
    def predict_evaluate(self, lead):
        lead = torch.unsqueeze(lead, dim=-2)
        y = self.model(lead.float())
        return y.argmax(dim=1)
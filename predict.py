import torch
from model import RNN, MLP
import torch.nn.functional as F
from data_preprocess import cal_x
import numpy as np


class Classifier():
    def __init__(self, path, device = 'cpu'):
        self.len_leads = 1000
        self.device = device
        self.num_classes = 9
        self.model = RNN(self.len_leads, self.num_classes)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        self.rnn1 = RNN(self.len_leads, self.num_classes)
        self.rnn1.load_state_dict(torch.load('rnn1.ckpt'))
        self.rnn2 = RNN(self.len_leads, self.num_classes)
        self.rnn2.load_state_dict(torch.load('rnn2.ckpt'))
        self.mlp = MLP(9*2, 9)
        self.mlp.load_state_dict(torch.load('mlp.ckpt'))
        self.mlp.eval()
        self.rnn2.eval()
        self.rnn1.eval()
    def predict(self, mat):
        mat = mat['ECG'][0][0]
        sex = mat[0][0]
        age = mat[1][0][0]
        data = mat[2][3] #select lead
        #data = data[:self.len_leads]
        #print(data.shape)

        len_lead = data.shape[0]
        if len_lead>1000:
            m, n = cal_x(len_lead)
            check_point = 0
            y = []
            prob = 0
            for j in range(m):
                if j != m - 1:
                    data_temp = data[check_point:(check_point + 1000)]
                    check_point = check_point + 1000 - n
                else:
                    data_temp = data[len_lead - 1000:]
                #print(check_point)
                x = torch.from_numpy(data_temp)
                x = torch.unsqueeze(x, dim=0)
                x = torch.unsqueeze(x, dim=0)
                x = x.to(self.device)

                output = self.model(x.float())
                p = F.softmax(output, dim=0)
                prob = prob + p.detach().numpy()

                out = output.argmax(dim=1).tolist()
                y.append(out[0])
            #print(y)
            y = np.asarray(y)
            y = np.bincount(y)
            y = np.argmax(y)
            return [y, prob/self.num_classes]

        else:
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
    def predict_2(self, lead):
        mat = lead['ECG'][0][0]
        sex = mat[0][0]
        age = mat[1][0][0]
        data = mat[2][0] # select lead
        data2 = mat[2][3]
        # data = data[:self.len_leads]
        # print(data.shape)

        len_lead = data.shape[0]
        if len_lead > 1000:
            m, n = cal_x(len_lead)
            check_point = 0
            y = []
            prob = 0
            for j in range(m):
                if j != m - 1:
                    data_temp1 = data[check_point:(check_point + 1000)]
                    data_temp2 = data2[check_point:(check_point + 1000)]
                    check_point = check_point + 1000 - n
                else:
                    data_temp = data[len_lead - 1000:]
                # print(check_point)
                x1 = torch.from_numpy(data_temp1)
                x1 = torch.unsqueeze(x1, dim=0)
                x1 = torch.unsqueeze(x1, dim=0)
                x1 = x1.to(self.device)

                x2 = torch.from_numpy(data_temp2)
                x2 = torch.unsqueeze(x2, dim=0)
                x2 = torch.unsqueeze(x2, dim=0)
                x2 = x2.to(self.device)

                out1 = self.rnn1(x1.float())
                out2 = self.rnn2(x2.float())
                out = torch.cat((out1, out2), dim=1)
                output = self.mlp(out)

                p = F.softmax(output, dim=0)
                prob = prob + p.detach().numpy()

                out = output.argmax(dim=1).tolist()
                y.append(out[0])
            # print(y)
            y = np.asarray(y)
            y = np.bincount(y)
            y = np.argmax(y)
            return [y, prob / self.num_classes]

        else:
            x = torch.from_numpy(data)
            x = torch.unsqueeze(x, dim=0)
            x = torch.unsqueeze(x, dim=0)
            x = x.to(self.device)

            y = self.model(x.float())
            y = F.softmax(y)
            return y

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RNN(nn.Module):
    def __init__(self, input_size, num_classes, device = 'cpu'):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = 512
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        # initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = RNN(4000, 1024, 1, 2)
    net.load_state_dict(torch.load('model.ckpt'))
    net.eval()
    print(net)
    from scipy import io as io

    mat = io.loadmat('datasets_test/test/1/A0229.mat')

    mat = mat['ECG'][0][0]

    sex = mat[0][0]
    age = mat[1][0][0]
    data = mat[2][0]
    data = data[:4000]
    x = torch.from_numpy(data)

    x = torch.unsqueeze(x, dim=0)
    x = torch.unsqueeze(x, dim=0)
    x = x.to(device)

    y = net(x.float())
    print(F.softmax(y))

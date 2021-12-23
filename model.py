import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

T = 1

class CNN(nn.Module):
    def __init__(self, input_size = 800, device = 'cpu'):
        super(CNN, self).__init__()
        self.device = device
        self.layer_1 = nn.Sequential(
            nn.Conv2d(12, 36, kernel_size=(1, 5), padding=0, stride=1),
            nn.ReLU()
        )

class RNN(nn.Module):
    def __init__(self, input_size, num_classes, device = 'cpu'):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = 128
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(self.hidden_size, num_classes)
    def forward(self, x):
        # initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        #print(x.size(0))

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        #out = self.fc2(out)
        #print(out.shape)
        return out
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        return y

if __name__ == '__main__':
    rnn1 = RNN(1000, 128)
    rnn2 = RNN(1000, 128)
    rnn3 = RNN(1000, 128)

    mlp = MLP(128*3, 9)

    x = torch.rand(3,12,1000)

    print(x[:,0:1,:].shape)

    out1 = rnn1(x[:, 0:1, :])
    out2 = rnn2(x[:, 1:2, :])
    out3 = rnn3(x[:, 2:3, :])

    out = torch.cat((out1, out2, out3), dim=1)
    out = mlp(out)
    print(out.shape)




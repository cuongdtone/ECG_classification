import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from model import RNN, MLP
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
from glob import glob
from sklearn.model_selection import train_test_split
from plot import plot
import numpy as np
from torchvision import transforms


input_size = 800
epochs = 50
batch_size = 32

class dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    # load an one of images
    def __getitem__(self, idx):
        leads_path = self.file_list[idx]
        leads = io.loadmat(leads_path)
        mat = leads['ECG']
        #sex = mat[0][0]
        #age = mat[1][0][0]
        data = mat
        #data = data[0][:input_size] #test no split
        #data = np.asarray(data, dtype=np.double)
        #data = (data + 4)/8
        #data = self.transform(data)
        #data = data.float()

        label = leads_path.split('/')[-2]
        label = int(label) - 1
        return data, label

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_path = '/home/cuong/Desktop/ECG/datasets_processing/'

    datasets_list = glob(input_path + '/*/*.mat' )
    train_list, val_list = train_test_split(datasets_list, test_size=0.2)

    train_data = dataset(train_list)
    val_data = dataset(val_list)

    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=True)

    print(len(train_data), len(train_loader))
    print(len(val_data), len(val_loader))
    print(train_data[0][0].shape)

    rnn1 = RNN(input_size, 1)
    rnn2 = RNN(input_size, 1)
    rnn3 = RNN(input_size, 1)
    rnn4 = RNN(input_size, 1)
    rnn5 = RNN(input_size, 1)
    rnn6 = RNN(input_size, 1)
    rnn7 = RNN(input_size, 1)
    rnn8 = RNN(input_size, 1)
    rnn9 = RNN(input_size, 1)
    rnn10 = RNN(input_size, 1)
    rnn11 = RNN(input_size, 1)
    rnn12 = RNN(input_size, 1)

    rnn1.train()
    rnn2.train()
    rnn3.train()
    rnn4.train()
    rnn5.train()
    rnn6.train()
    rnn7.train()
    rnn8.train()
    rnn9.train()
    rnn10.train()
    rnn11.train()
    rnn12.train()

    optimizer1 = torch.optim.Adam(rnn1.parameters(), lr=0.01)
    optimizer2 = torch.optim.Adam(rnn1.parameters(), lr=0.01)
    optimizer3 = torch.optim.Adam(rnn1.parameters(), lr=0.01)
    optimizer4 = torch.optim.Adam(rnn1.parameters(), lr=0.01)
    optimizer5 = torch.optim.Adam(rnn1.parameters(), lr=0.01)
    optimizer6 = torch.optim.Adam(rnn1.parameters(), lr=0.01)
    optimizer7 = torch.optim.Adam(rnn1.parameters(), lr=0.01)
    optimizer8 = torch.optim.Adam(rnn1.parameters(), lr=0.01)
    optimizer9 = torch.optim.Adam(rnn1.parameters(), lr=0.01)
    optimizer10 = torch.optim.Adam(rnn1.parameters(), lr=0.01)
    optimizer11 = torch.optim.Adam(rnn1.parameters(), lr=0.01)
    optimizer12 = torch.optim.Adam(rnn1.parameters(), lr=0.01)

    mlp = MLP(1*12, 9)
    mlp.train()
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print('Epoch %d:'%(epoch+1))
        for i, (lead, labels) in enumerate(train_loader):
            #lead = torch.unsqueeze(lead, dim=-2)
            #print(lead.shape)
            lead = lead.to(device)
            labels = labels.to(device)

            # Forward pass
            out1 = rnn1(lead[:, 0:1, :].float())
            out2 = rnn1(lead[:, 1:2, :].float())
            out3 = rnn1(lead[:, 2:3, :].float())
            out4 = rnn1(lead[:, 3:4, :].float())
            out5 = rnn1(lead[:, 4:5, :].float())
            out6 = rnn1(lead[:, 5:6, :].float())
            out7 = rnn1(lead[:, 6:7, :].float())
            out8 = rnn1(lead[:, 7:8, :].float())
            out9 = rnn1(lead[:, 8:9, :].float())
            out10 = rnn1(lead[:, 9:10, :].float())
            out11 = rnn1(lead[:, 10:11, :].float())
            out12 = rnn1(lead[:, 11:, :].float())
            #print('out1: ', out1.shape)

            out = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12], dim=1)
            #print('out: ',out.shape)
            outputs = mlp(out)
            #print(outputs.shape)
            #print(labels.shape)


            loss = criterion(outputs, labels)


            # Backward and optimize
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            optimizer5.zero_grad()
            optimizer6.zero_grad()
            optimizer7.zero_grad()
            optimizer8.zero_grad()
            optimizer9.zero_grad()
            optimizer10.zero_grad()
            optimizer11.zero_grad()
            optimizer12.zero_grad()
            optimizer_mlp.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()
            optimizer6.step()
            optimizer7.step()
            optimizer8.step()
            optimizer9.step()
            optimizer10.step()
            optimizer11.step()
            optimizer12.step()
            optimizer_mlp.step()


            if (i + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, len(train_loader), loss.item()))
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for lead, labels in val_loader:
                #lead = torch.unsqueeze(lead, dim=-2)
                #print(lead.shape)
                lead = lead.to(device)
                labels = labels.to(device)

                out1 = rnn1(lead[:, 0:1, :].float())
                out2 = rnn1(lead[:, 1:2, :].float())
                out3 = rnn1(lead[:, 2:3, :].float())
                out4 = rnn1(lead[:, 3:4, :].float())
                out5 = rnn1(lead[:, 4:5, :].float())
                out6 = rnn1(lead[:, 5:6, :].float())
                out7 = rnn1(lead[:, 6:7, :].float())
                out8 = rnn1(lead[:, 7:8, :].float())
                out9 = rnn1(lead[:, 8:9, :].float())
                out10 = rnn1(lead[:, 9:10, :].float())
                out11 = rnn1(lead[:, 10:11, :].float())
                out12 = rnn1(lead[:, 11:, :].float())
                # print('out1: ', out1.shape)

                out = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12], dim=1)
                # print('out: ',out.shape)
                outputs = mlp(out)
                # print(outputs.shape)
                # print(labels.shape)

                val_loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                acc = ((outputs.argmax(dim=1) == labels).float().mean())
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

        print('Epoch [{}/{}], val_accuracy : {}, val_loss : {}'.format(epoch + 1, epochs, epoch_val_accuracy, epoch_val_loss))
        print('')

    torch.save(rnn1.state_dict(), 'rnn1.ckpt')
    torch.save(rnn2.state_dict(), 'rnn2.ckpt')
    torch.save(rnn3.state_dict(), 'rnn3.ckpt')
    torch.save(rnn4.state_dict(), 'rnn4.ckpt')
    torch.save(rnn5.state_dict(), 'rnn5.ckpt')
    torch.save(rnn6.state_dict(), 'rnn6.ckpt')
    torch.save(rnn7.state_dict(), 'rnn7.ckpt')
    torch.save(rnn8.state_dict(), 'rnn8.ckpt')
    torch.save(rnn9.state_dict(), 'rnn9.ckpt')
    torch.save(rnn10.state_dict(), 'rnn10.ckpt')
    torch.save(rnn11.state_dict(), 'rnn11.ckpt')
    torch.save(rnn12.state_dict(), 'rnn12.ckpt')

    torch.save(mlp.state_dict(), 'mlp.ckpt')

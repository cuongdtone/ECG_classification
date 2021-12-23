import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from model import RNNs
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
from glob import glob
from sklearn.model_selection import train_test_split
from plot import plot
import numpy as np
from torchvision import transforms


input_size = 1000
epochs = 20
batch_size = 64

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

    rnn = RNNs()
    rnn.train()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print('Epoch %d:'%(epoch+1))
        for i, (lead, labels) in enumerate(train_loader):
            #lead = torch.unsqueeze(lead, dim=-2)
            #print(lead.shape)
            lead = lead.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs= rnn(lead)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 250 == 0:
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

                outputs = rnn(lead)
                loss = criterion(outputs, labels)

                val_loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)

                acc = ((outputs.argmax(dim=1) == labels).float().mean())
                epoch_val_accuracy += acc / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

        print('Epoch [{}/{}], val_accuracy : {}, val_loss : {}'.format(epoch + 1, epochs, epoch_val_accuracy, epoch_val_loss))
        print('')

    torch.save(rnn.state_dict(), 'rnns.ckpt')


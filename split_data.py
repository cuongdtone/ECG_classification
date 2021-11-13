import scipy.io as io
import matplotlib.pyplot as plt
from glob import glob
import os
import shutil
from torch.utils.data import DataLoader


def plot(data, pathsave=''):
    plt.figure(figsize=(30,18))
    plt.title()
    for i in range(12):
        plt.subplot(2,6,i+1)
        plt.axis([0, 3000, -2, 2])
        plt.plot(data[i],  linewidth=0.5)
    plt.savefig(pathsave)
    #plt.show()


file  = open('/home/cuong/Desktop/ECG/REFERENCE.csv')
labels = file.readlines()[1:]
file.close()


input_path = '/home/cuong/Desktop/ECG/TrainingSet1/'

save_path = 'datasets/'


for i in labels:
    i = i.rstrip('\n')
    i = i.split(',')

    data_file_name = i[0]
    first_label = i[1]
    second_label = i[2]
    third_label = i[3]

    if second_label == '':
        path = save_path + first_label
        if not os.path.exists(path):
            os.mkdir(path)
        data_path = input_path + '/' + data_file_name + '.mat'

        if os.path.exists(data_path):
            shutil.copy(data_path, path)
    elif third_label == '':
        path = save_path + first_label + '_' + second_label
        if not os.path.exists(path):
            os.mkdir(path)
        data_path = input_path + '/' + data_file_name + '.mat'
        if os.path.exists(data_path):
            shutil.copy(data_path, path)
    else:
        path = save_path + first_label + '_' + second_label + '_' + third_label
        if not os.path.exists(path):
            os.mkdir(path)
        data_path = input_path + '/' + data_file_name + '.mat'
        if os.path.exists(data_path):
            shutil.copy(data_path, path)






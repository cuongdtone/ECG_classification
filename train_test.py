from glob import glob
from sklearn.model_selection import train_test_split
import shutil
import os

index = 9
datasets = '/home/cuong/Desktop/ECG/datasets'

train = '/home/cuong/Desktop/ECG/datasets_split/train'
test = '/home/cuong/Desktop/ECG/datasets_split/test'


for i in range(0,index):
    i = i+1
    print(i)
    list_mat = glob(datasets + '/%d'%(i) + '/*.mat')

    train_list, test_list = train_test_split(list_mat, test_size=0.1)
    os.mkdir(train + '/%d'%(i))
    os.mkdir(test + '/%d'%(i))
    for scr in train_list:
        shutil.copy(scr, train + '/%d'%(i))
    for scr in test_list:
        shutil.copy(scr, test + '/%d'%(i))

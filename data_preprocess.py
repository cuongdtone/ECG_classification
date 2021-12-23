from scipy import io as io
import math
from glob import glob
import os

len_lead = 800

def cal_x(data_length, len_lead=len_lead):
    m = data_length/1000
    m = int(math.ceil(m)*1.5)
    x = (len_lead*m - data_length)/(m-1)
    return m, int(x) #m: m seq, x: overlap len

if __name__ == '__main__':
    num_class = 9
    for index_class in range(1, num_class+1):
        input_path = '/home/cuong/Desktop/ECG/datasets_split/train/%d'%(index_class)
        save_path = '/home/cuong/Desktop/ECG/datasets_processing/%d'%(index_class)
        os.mkdir(save_path)

        list_mat = glob(input_path + '/*.mat')
        for i in list_mat:
            mat = io.loadmat(i)
            mat = mat['ECG'][0][0]

            sex = mat[0][0]
            age = mat[1][0][0]
            data = mat[2]
            data_length = len(data.T)
            m, x = cal_x(data_length)
            #print(data_length, m, x)
            check_point = 0
            for j in range(m):
                if j!=m-1:
                    data_temp = data[:, check_point:check_point+len_lead]
                    check_point = check_point + len_lead - x
                else:
                    data_temp = data[:, data_length-len_lead:]
                name = i.split('/')[-1].split('.')[0]
                name = name + '_%02d'%(j) + '.mat'
                path = save_path + '/' + name
                #print(name)
                #print(data_temp.shape)
                data_write = {'ECG': data_temp}
                io.savemat(path, data_write)


            #print('  ')
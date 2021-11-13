import scipy.io as io
import matplotlib.pyplot as plt
from glob import glob


def plot(data, pathsave=''):
    plt.figure(figsize=(30,18))
    plt.title(sex)
    for i in range(12):
        plt.subplot(2, 6,i+1)
        plt.axis([0, 3000, -2, 2])
        plt.plot(data[i],  linewidth=0.5)
    plt.savefig(pathsave)
    plt.show()

if __name__ == '__main__':
    input_path = 'datasets/3'
    list_mat = glob(input_path + '/*.mat')
    list_mat.sort()

    length = []
    for i in list_mat:
        mat = io.loadmat(i)
        mat = mat['ECG'][0][0]

        sex = mat[0][0]
        age = mat[1][0][0]
        data = mat[2]

        length.append(len(data.T))
        print(len(data.T))
        plot(data)

    print(max(length))
    print(min(length))





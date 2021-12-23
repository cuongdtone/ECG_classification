import scipy.io as io
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import warnings
from sklearn.metrics import accuracy_score


def plot_cm(cm, num_classes=9, accuracy=0, normalize=True, save_dir='', names=()):
    try:
        import seaborn as sn

        array = cm / ((cm.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        plt.title('Accuracy: %.2f %s '%(accuracy*100, '%'))
        sn.set(font_scale=1.0 if num_classes < 50 else 0.8)  # for label size
        labels = (0 < len(names) < 99) and len(names) == num_classes  # apply names to ticklabels
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array, annot=num_classes < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
        fig.axes[0].set_xlabel('True')
        fig.axes[0].set_ylabel('Predicted')
        fig.savefig(save_dir + 'confusion_matrix.png', dpi=250)
    except Exception as e:
        print(f'WARNING: ConfusionMatrix plot failure: {e}')
def plot(data, pathsave=''):
    plt.figure(figsize=(30,18))
    #plt.title(sex)
    for i in range(12):
        plt.subplot(2, 6,i+1)
        plt.axis([0, 800, -2, 2])
        plt.plot(data[i],  linewidth=0.5)
    plt.savefig(pathsave)
    plt.show()

def scores(CM):
    F1x = []
    for i in range(9):
        F = 2 * CM[i,i]/(np.sum([CM[:,i], CM[i,:]]))
        F1x.append(F)
    F1 = np.sum(F1x)/9
    return F1, F1x

if __name__ == '__main__':
    input_path = '/home/cuong/Desktop/ECG/CPSC'
    list_mat = glob(input_path + '/*.mat')
    #list_mat.sort()

    length = []
    for i in list_mat:
        mat = io.loadmat(i)
        mat = mat['ECG']
        data = mat

        length.append(len(data.T))
        print(len(data.T))
        plot(data)

    print(max(length))
    print(min(length))





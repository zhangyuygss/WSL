"""
Plot figures from npy or logging files saved while training
================================================
*Author*: Yu Zhang, Northwestern Polytechnical University
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import sys


def plot_from_npy(npyfile):
    assert os.path.isfile(npyfile)
    info = np.load(npyfile).item()
    tr_loss = info['train_loss']
    tr_loss_detail = info['train_loss_detail']
    tr_acc = info['train_accuracy']
    val_loss = info['val_loss']
    val_acc = info['val_accuracy']
    epochs = len(tr_loss)
    x = np.linspace(1, epochs, epochs)
    xx = np.linspace(1, epochs, len(tr_loss_detail))
    plt.plot(x, tr_loss, ls='-', color='r')
    plt.plot(x, tr_acc, ls='-.', color='r')
    plt.plot(x, val_loss, ls='-', color='b')
    plt.plot(x, val_acc, ls='-.', color='b')
    plt.plot(xx, tr_loss_detail, ls='-', color='y')
    plt.legend(labels=['tr_loss', 'tr_accuracy', 'val_loss',
                       'val_accuracy', 'tr_detail_loss'], loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss or accuracy')
    plt.title('training loss and accuracy')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Please input at least one log file for plotting')
    else:
        for log_file in sys.argv[1:]:
            plot_from_npy(log_file)


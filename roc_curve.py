import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp

if __name__ == "__main__":

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    list = ["lstm", "lstm_smote", "lstm_weighted", "cnn_lstm"]

    for name in list:
        y_score = np.loadtxt(name + "_score")
        y = np.loadtxt(name + "_label")
        print()

        fpr[name], tpr[name], _ = roc_curve(y[:, 1], y_score[:, 1])
        roc_auc[name] = auc(fpr[name], tpr[name])

    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    lw = 2

    plt.plot(fpr["lstm"], tpr["lstm"],
             label='LSTM ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["lstm"]),
             color='cornflowerblue', linestyle=':', linewidth=4)

    plt.plot(fpr["lstm_smote"], tpr["lstm_smote"],
             label='LSTM with SMOTE ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["lstm_smote"]),
             color='darkorange', linestyle=':', linewidth=4)

    plt.plot(fpr["lstm_weighted"], tpr["lstm_weighted"],
             label='LSTM with weighted loss func ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["lstm_weighted"]),
             color='aqua', linestyle=':', linewidth=4)

    plt.plot(fpr["cnn_lstm"], tpr["cnn_lstm"],
             label='CNN-LSTM func ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["cnn_lstm"]),
             color='red', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for each model')
    plt.legend(loc="lower right")
    plt.show()
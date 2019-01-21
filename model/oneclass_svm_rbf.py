'''
2019. 1. 19
One-class SVM

'''
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

class Model(object):
    def __init__(self, debug=False):
        self.model = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        self.debug = debug
        self.reg = None
        return

    def train(self, data, labels, hyperparams = None):
        '''
        data : list of input
        labels : list of corresponding output
        
        return : None 
        '''
        # only using data with label 1(abnormal)
        oneclass_data = data[labels == 0]
        if self.debug:
            print("total data size: {}".format(len(data)))
            print("abnormal data size: {}".format(len(oneclass_data)))
        self.model.fit(oneclass_data)
        if self.debug:
            print(self.reg.score(data, labels))
        return 

    def test(self, data, labels):
        '''
        data : list of input

        return : list of predicted output
        '''
        labels_pred = self.model.predict(data)
        labels_pred = [0 if i > 0.5 else 1 for i in labels_pred]

        return confusion_matrix(labels, labels_pred) 


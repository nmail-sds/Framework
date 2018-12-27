# sgd classifier

'''
2018. 12. 27
SGD-based classifier

'''

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix

class Model(object):
    def __init__(self, debug=False):
        self.model = SGDClassifier()
        self.debug = debug
        self.reg = None
        return

    def train(self, data, labels):
        '''
        data : list of input
        labels : list of corresponding output
        
        return : None 
        '''
        if self.debug:
            print(data.shape)
            print(labels.shape)
        self.model.fit(data, labels)
        return 

    def test(self, data, labels):
        '''
        data : list of input

        return : list of predicted output
        '''
        labels_pred = self.model.predict(data)
        return confusion_matrix(labels, labels_pred) 

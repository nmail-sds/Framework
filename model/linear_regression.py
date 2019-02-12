# Linear Regression

'''
2018. 11. 22
선형 회귀 기법을 이용한 데이터 분류 모델

* Scikit-learn을 wrap하여 생성
'''

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix 

class Model(object):
    def __init__(self, debug=False):
        self.model = LinearRegression()
        self.debug = debug
        self.reg = None
        return

    def train(self, data, labels, hyperparams = None):
        '''
        data : list of input
        labels : list of corresponding output
        
        return : None 
        '''
        if self.debug:
            print(data.shape)
            print(labels.shape)
        if len(data.shape) > 2:
            data = data.reshape([data.shape[0], -1])
        self.reg = self.model.fit(data, labels)
        if self.debug:
            print(self.reg.score(data, labels))
        return 

    def test(self, data, labels):
        '''
        data : list of input

        return : list of predicted output
        '''
        if len(data.shape) > 2:
            data = data.reshape([data.shape[0], -1])
        labels_pred = np.around(self.reg.predict(data)).astype(int)
        return confusion_matrix(labels, labels_pred, labels=[0, 1])
    

# Fully-connected Neural Network

'''
2018. 12. 26
단순 연결 신경망 모델

'''

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense

class Model(object):

    def __init__(self, debug=False):
    
        model = Sequential([
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(2, activation="softmax"),
            ])
        
        model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])
        
        self.model = model
        return

    def train(self, data, labels, hyperparams = dict()):
        '''
        data : list of input
        labels : list of corresponding output
        hyperparams = dictionary maps from name to its value
        return : None 
        '''
        binary_labels = np.asarray(list(map(lambda i: [0, 1] if i else [1, 0], labels)))
        self.model.fit(data, binary_labels, epochs=30, batch_size=32)
        
        return 

    def test(self, data, labels):
        '''
        data : list of input

        return : list of predicted output
        '''
        binary_labels = np.asarray(list(map(lambda i: [0, 1] if i else [1, 0], labels)))
        return self.model.evaluate(data, binary_labels, batch_size = 128)


# Fully-connected Neural Network

'''
2018. 12. 26
단순 연결 신경망 모델

'''

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam

from loss import weighted_categorical_crossentropy 

class Model(object):

    def __init__(self, debug=False):
    
        model = Sequential([
            #Flatten(),
            Dense(32, activation="relu"),
            Dense(8, activation="relu"),
            Dense(4, activation="relu"),
            Dense(2, activation="softmax"),
            ])
        
        self.model = model
        return
    
    def _int_to_binary(self, labels):
        return np.asarray(list(map(lambda i: [0, 1] if i == 1 else [1, 0], labels)))

    def train(self, data, labels, hyperparams = dict()):
        '''
        data : list of input
        labels : list of corresponding output
        hyperparams = dictionary maps from name to its value
        
        return : None 
        '''

        # loss count 
        one_occurence = np.count_nonzero(labels)
        zero_occurence = len(labels) - one_occurence
        self.alpha = zero_occurence / one_occurence
        
        self.model.compile(optimizer=Adam(),
                loss=weighted_categorical_crossentropy(self.alpha),
                metrics=['accuracy'])

        binary_labels = self._int_to_binary(labels)
        if "validation" in hyperparams.keys():
            val_data = hyperparams["validation"][0]
            val_labels = self._int_to_binary(hyperparams["validation"][1])
            self.model.fit(data, binary_labels, epochs=200, batch_size=32, 
                    validation_data = (val_data, val_labels))
        else:
            self.model.fit(data, binary_labels, epochs=200, batch_size=32)
        return 

    def test(self, data, labels):
        '''
        data : list of input

        return : list of predicted output
        '''
        binary_labels = np.asarray(list(map(lambda i: [0, 1] if i else [1, 0], labels)))
        labels_pred = self.model.predict(data, batch_size = 128)
        labels_pred = [0 if l[0] > l[1] else 1 for l in labels_pred]
        
        return confusion_matrix(labels, labels_pred)

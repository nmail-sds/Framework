# 1D Convolutional Neural Network

'''
2018. 12. 27
1차원 합성곱 신경망 모델

Only works for sequential data
dataset = ["earthquake"]
'''

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Reshape, Dense, Flatten
from keras.losses import binary_crossentropy
from sklearn.metrics import confusion_matrix

#loss function options
import keras.backend as K 
from model.loss import weighted_categorical_crossentropy

class Model(object):

    def __init__(self, debug=False):
        self.alpha = 1.0
        self.debug = debug
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

        if self.debug:
            print("data shape: " + str(data.shape))
        
        model = Sequential([
            Reshape((512, 1),input_shape = (512,)),
            
            Conv1D(20, kernel_size=5, activation="relu", padding="valid"),
            MaxPooling1D(pool_size=2),
            
            Conv1D(40, kernel_size=5, activation="relu", padding="valid"),
            MaxPooling1D(pool_size=2),
            
            Conv1D(40, kernel_size=5, activation="relu", padding="valid"),
            MaxPooling1D(pool_size=2),
            
            Conv1D(80, kernel_size=5, activation="relu", padding="valid"),
            MaxPooling1D(pool_size=2),
             
            Conv1D(80, kernel_size=5, activation="relu", padding="valid"),
            MaxPooling1D(pool_size=2),
              
            Conv1D(160, kernel_size=5, activation="relu", padding="valid"),
            MaxPooling1D(pool_size=2),
           
            Flatten(),
            
            Dense(32, activation = "relu"),
            Dense(2, activation="softmax"),
            ])
        
        model.compile(optimizer='adam',
                loss=weighted_categorical_crossentropy(self.alpha),
                metrics=['accuracy'])
        
        model.build(data.shape[1:])
        model.summary()
        self.model = model
        
        binary_labels = np.asarray(list(map(lambda i: [0, 1] if i else [1, 0], labels)))
        
        if "validation" in hyperparams.keys():
            val_data = hyperparams["validation"][0]
            val_labels = self._int_to_binary(hyperparams["validation"][1])
            self.model.fit(data, binary_labels, epochs=20, batch_size=32, 
                    validation_data = (val_data, val_labels))
        else:
            self.model.fit(data, binary_labels, epochs=20, batch_size=32)
        
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

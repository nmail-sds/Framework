'''
2019. 1. 21
Data structure of dataset

Yihan Kim
'''

# Definitions of dataset container 

class Pair(object):
    def __init__(self, data, labels):
        self.data = data 
        self.labels = labels

class Data(object):
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train = Pair(train_data, train_labels)
        self.test = Pair(test_data, test_labels)

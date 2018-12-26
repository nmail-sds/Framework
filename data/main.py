'''
data/read.py
2018. 11. 22

데이터셋 이름을 입력받아 데이터를 불러오는 기능

'''

import os
import csv
import importlib
import numpy as np
from scipy.io import arff

# util - absolute directory of current file 
dir_path = os.path.dirname(os.path.realpath(__file__))

class Pair(object):
    def __init__(self, data, labels):
        self.data = data 
        self.labels = labels

class Data(object):
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train = Pair(train_data, train_labels)
        self.test = Pair(test_data, test_labels)

# primitive function that read data file

def _read_csv(filename, basedir="."):
    filedir = os.path.join(basedir, filename)
    with open(filedir, 'r') as csv_f:
        data = list(csv.reader(csv_f))
    for i, row in enumerate(data):
        for j, item in enumerate(row):
            data[i][j] = '0.0' if item == '' else item
    return data

def _read_arff(filename, basedir="."):
    filedir = os.path.join(basedir, filename)
    data, meta = arff.loadarff(filedir)
    return data

# wrapping function that handles specific dataset

def _read_secom():
    # unsplitted data
    # 1566 -> 500 + 866
    working_dir = os.path.join(dir_path, "uci-secom")
    
    uci_secom_data = []
    for filename in os.listdir(working_dir):
        uci_secom_data.extend(_read_csv(filename, working_dir))
    uci_secom_data = np.asarray(uci_secom_data)
    #print(uci_secom_data.shape)

    # secom data has time column, so we will erase it
    train_data = uci_secom_data[:500, 1:-1]
    test_data = uci_secom_data[500:, 1:-1]
    train_labels = uci_secom_data[:500, -1]
    test_labels = uci_secom_data[500:, -1]

    return Data(train_data, train_labels, test_data, test_labels)

def _read_wafer():
    # train data 6164 * 152
    # test data 1000 * 152
    # train labels 6164 
    # test labels 1000 

    working_dir = os.path.join(dir_path, "wafer")
    wafer_data = {}
    for filename in os.listdir(working_dir):
        if ".arff" in filename:
            # there are some other formatted data(e.g. txt or md), so filter
            if "TEST" in filename:
                # add test data
                wafer_data["train"] = _read_arff(filename, working_dir)
            if "TRAIN" in filename:
                # add train data
                wafer_data["test"] = _read_arff(filename, working_dir)

    train_data = np.asarray(list(map(lambda x: list(x)[:-1], wafer_data["train"])))
    test_data = np.asarray(list(map(lambda x: list(x)[:-1], wafer_data["test"])))
    train_labels = (np.asarray(list(map(lambda x: int(x[-1]), wafer_data["train"]))) + 1) / 2
    test_labels = (np.asarray(list(map(lambda x: int(x[-1]), wafer_data["test"]))) + 1) / 2

    return Data(train_data, train_labels, test_data, test_labels)

def main(dataset: str):
    if dataset == "uci-secom":
        secom_data = _read_secom()
        return secom_data

    if dataset == "wafer":
        wafer_data = _read_wafer()
        return wafer_data

    if dataset == "etc":
        # add something in here
        return None

if __name__ == "__main__":
    main()


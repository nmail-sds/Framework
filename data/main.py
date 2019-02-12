'''
data/read.py
2018. 11. 22

데이터셋 이름을 입력받아 데이터를 불러오는 기능

'''

import os
import sys
import csv
import importlib
import numpy as np
from scipy.io import arff
import pickle_loader as pkl_loader

from ds import Pair, Data
from sampling import smote_dataset
# util - absolute directory of current file 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, ".."))


# Primitive function that read data file

def _read_csv(filename, basedir="."):
    filedir = os.path.join(basedir, filename)
    with open(filedir, 'r') as csv_f:
        data = list(csv.reader(csv_f))
    # handles the data that have null column
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
    # 1568 -> 1199 + 368
    working_dir = os.path.join(dir_path, "uci-secom")
    
    uci_secom_data = []
    for filename in os.listdir(working_dir):
        uci_secom_data.extend(_read_csv(filename, working_dir))
    uci_secom_data = np.asarray(uci_secom_data)
    # first line has label name, so it will be removed.
    # secom data has time column(at first), so we will erase it
    
    train_data = uci_secom_data[1:1200, 1:-1].astype(float)
    test_data = uci_secom_data[1200:, 1:-1].astype(float)
    train_labels = (uci_secom_data[1:1200, -1].astype(int) + 1) / 2
    test_labels = (uci_secom_data[1200:, -1].astype(int) + 1) / 2

    return Data(train_data, train_labels, test_data, test_labels)

def _read_secom_preprocessed(process_type: str = None):
    working_dir = os.path.join(dir_path, "uci-secom-preprocessed")
    
    data = Data(None, None, None, None)
    
    for filename in os.listdir(working_dir):
        # common : labels
        if filename == "train.labels.csv":
            data.train.labels = np.asarray(_read_csv(filename, working_dir)).astype(int).flatten()
        if filename == "test.labels.csv":
            data.test.labels = np.asarray(_read_csv(filename, working_dir)).astype(int).flatten()

        if process_type == None:
            if filename == "train_knnImpute.csv":
                data.train.data = np.asarray(_read_csv(filename, working_dir)).astype(float)
            if filename == "test_knnImpute.csv":
                data.test.data = np.asarray(_read_csv(filename, working_dir)).astype(float)

        if process_type == "pca":
            if filename == "train_pca.csv":
                data.train.data = np.asarray(_read_csv(filename, working_dir)).astype(float)
            if filename == "test_pca.csv":
                data.test.data = np.asarray(_read_csv(filename, working_dir)).astype(float)

        if process_type == "ica":
            if filename == "train_ica.csv":
                data.train.data = np.asarray(_read_csv(filename, working_dir)).astype(float)
            if filename == "test_ica.csv":
                data.test.data = np.asarray(_read_csv(filename, working_dir)).astype(float)

        if process_type == "chisq":
            if filename == "train_chisq.csv":
                data.train.data = np.asarray(_read_csv(filename, working_dir)).astype(float)
            if filename == "test_chisq.csv":
                data.test.data = np.asarray(_read_csv(filename, working_dir)).astype(float)
 
    return data


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
                wafer_data["test"] = _read_arff(filename, working_dir)
            if "TRAIN" in filename:
                # add train data
                wafer_data["train"] = _read_arff(filename, working_dir)

    train_data = np.asarray(list(map(lambda x: list(x)[:-1], wafer_data["train"])))
    test_data = np.asarray(list(map(lambda x: list(x)[:-1], wafer_data["test"])))
    train_labels = (1 - np.asarray(list(map(lambda x: int(x[-1]), wafer_data["train"])))) / 2
    test_labels = (1 - np.asarray(list(map(lambda x: int(x[-1]), wafer_data["test"])))) / 2

    return Data(train_data, train_labels, test_data, test_labels)

def _read_earthquakes():
    # train_data 322 * 512
    # test_data 139 * 512
    # train_labels 322
    # test_labels 139

    working_dir = os.path.join(dir_path, "earthquakes")
    wafer_data = {}
    for filename in os.listdir(working_dir):
        if ".arff" in filename:
            # there are some other formatted data(e.g. txt or md), so filter
            if "TEST" in filename:
                # add test data
                wafer_data["test"] = _read_arff(filename, working_dir)
            if "TRAIN" in filename:
                # add train data
                wafer_data["train"] = _read_arff(filename, working_dir)

    train_data = np.asarray(list(map(lambda x: list(x)[:-1], wafer_data["train"])))
    test_data = np.asarray(list(map(lambda x: list(x)[:-1], wafer_data["test"])))
    train_labels = np.asarray(list(map(lambda x: int(x[-1]), wafer_data["train"]))) 
    test_labels = np.asarray(list(map(lambda x: int(x[-1]), wafer_data["test"]))) 

    return Data(train_data, train_labels, test_data, test_labels)

'''
def _read_cmu_wafer():
    
    normal_dir = os.path.join(dir_path, "cmu-wafer", "normal")
    abnormal_dir = os.path.join(dir_path, "cmu-wafer", "abnormal")

    def parse_filename(filename: str):
        return filename.split(".")

    def dtoi(desc: str):
        return ('6', '7', '8', '11', '12', '15').index(desc)
    
    def read_file(filedir):
        with open(filedir, 'r') as f:
            r = csv.reader(f, delimiter='\t')
            try:
                return [line[1] for line in r]
            except:
                print(filedir)
                return
    
    def read_abnormal():
        data = {}
        labels = {}
        for filename in os.listdir(abnormal_dir):
            filedir = os.path.join(abnormal_dir, filename)
            run_wafer, desc = parse_filename(filename)
            if desc not in ('6', '7', '8', '11', '12', '15'):
                continue
            if not run_wafer in data.keys():
                data[run_wafer] = [None] * 6
            data[run_wafer][dtoi(desc)] = read_file(filedir)
            labels[run_wafer] = 1
        return data, labels

    def read_normal():
        data = {}
        labels = {}
        for filename in os.listdir(normal_dir):
            filedir = os.path.join(normal_dir, filename)
            run_wafer, desc = parse_filename(filename)
            if desc not in ('6', '7', '8', '11', '12', '15'):
                continue
            if not run_wafer in data.keys():
                data[run_wafer] = [None] * 6
            data[run_wafer][dtoi(desc)] = read_file(filedir)
            labels[run_wafer] = 0
        return data, labels
    
    ab_data, ab_labels = read_abnormal()
    no_data, no_labels = read_normal()

    # merge normal & abnormal data
    data_dict = {**ab_data, **no_data}
    labels_dict = {**ab_labels, **no_labels}

    # integrity check
    assert data_dict.keys() == labels_dict.keys()

    data = []
    labels = []

    for key in sorted(data_dict.keys()):
        # truncate first 100 elements from each series
        data.append(np.asarray(data_dict[key])[:, :100])
        labels.append(np.asarray(labels_dict[key]))

    data = np.array(data)
    labels = np.reshape(labels, -1) 
    
    np.random.seed(0)
    x = np.arange(len(data))
    np.random.shuffle(x)
    data = data[x]
    labels = labels[x]
    
    train_data = data[:800]
    test_data = data[800:]
    train_labels = labels[:800]
    test_labels = labels[800:]

    return Data(train_data, train_labels, test_data, test_labels)
'''

def _read_cmu_wafer():
    # load faster using pickle 
    data = Data(None, None, None, None)
    data.train.data = pkl_loader.pkl2np("train_data.pkl").astype(int)
    data.train.labels = pkl_loader.pkl2np("train_labels.pkl").astype(int)
    data.test.data = pkl_loader.pkl2np("test_data.pkl").astype(int)
    data.test.labels = pkl_loader.pkl2np("test_labels.pkl").astype(int)
    return data 

    
# main function 

def main(dataset_name: str, smote: bool = False):
    if dataset_name == "uci-secom":
        dataset = _read_secom()

    if dataset_name == "wafer":
        dataset = _read_wafer()
    
    if dataset_name == "earthquake":
        dataset = _read_earthquakes()

    if dataset_name == "cmu-wafer":
        dataset = _read_cmu_wafer()
    
    if dataset_name == "etc":
        # add something in here
        return None
       
    if dataset_name == "uci-secom-prep":
        dataset = _read_secom_preprocessed()
    
    if dataset_name == "uci-secom-pca":
        dataset = _read_secom_preprocessed("pca")
    
    if dataset_name == "uci-secom-ica":
        dataset = _read_secom_preprocessed("ica")
    
    if dataset_name == "uci-secom-chisq":
        dataset = _read_secom_preprocessed("chisq")
    
    
    # data manipulation
    if smote:
        dataset = smote_dataset(dataset)
    
    return dataset
# for unit test

if __name__ == "__main__":
    data = main(input("dataset name(uci-secom, wafer): "), smote = True)
    print(data.train.data.shape)
    print(data.train.labels.shape)
    print(data.test.data.shape)
    print(data.test.labels.shape)

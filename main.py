'''
main.py
2018. 11. 22
Yihan Kim

사용법 : main.py --model [model_name] --data [data_name]

'''

import argparse
import os 
import importlib
import numpy as np
import data.pickle_loader as pkl_loader
import model.lstm as lstm

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import sys
sys.path.append("./data")


parser = argparse.ArgumentParser(description = "모델, 데이터 및 실행 모드 선택")

parser.add_argument("--model", type=str, default="lstm", help="select mode (e.g. linear_regression, dense, sgd_classifier)")
parser.add_argument("--dataset", type=str, default="cmu-wafer", help="select dataset (e.g. uci-secom, wafer)")
parser.add_argument("--smote", action="store_true", default=False, help="resample minority class using SMOTE")

args = parser.parse_args()
# args.model, args.dataset을 통해 입력값을 사용할 수 있음
print(args.smote)

# args.model을 불러오기

from model import *
def import_model():
    # args.model에 해당하는 object 리턴
    # model = [args.model].Model()
    model = eval("{}.Model()".format(args.model))
    #model = linear_regression.Model(debug=True)
    return model

from data import main as read_data
def import_dataset():
    try:
        return read_data.main(args.dataset, args.smote)
    except:
        raise

def main():
    model = import_model()
    dataset = import_dataset()
    print(model)
    print(dataset.train.data)
    print(dataset.train.labels)
    print(dataset.test.data)
    print(dataset.test.labels)
    model.train(dataset.train.data, dataset.train.labels)
    predict = model.test(dataset.test.data, dataset.test.labels)
    print("confusion matrix: ")
    print(predict)
    return    

def main_cv(n_splits: int = 10):
    '''
    cross-validation 
    입력을 n_splits 수로 나누어 validation set을 형성
    
    input: n_splits : int (default 10)

    해야 되는것 정리 : time-series 로 된 반도체 데이터 찾기(중요도 중상, 가능성 하)
    batch_norm, reguralization 추가 (중요도 중, 가능성 중)
    데이터 normalize (중요도 중, 가능성 중)


    
    '''
    #dataset = import_dataset()

    dataset = read_data.Data(pkl_loader.pkl2np("train_data.pkl"), pkl_loader.pkl2np("train_labels.pkl"), pkl_loader.pkl2np("test_data.pkl"), pkl_loader.pkl2np("test_labels.pkl"))
    #kf = KFold(n_splits = n_splits)
    
    result = []

    #print(dataset.train.data.shape)
    #assert False

    #standarization
    train_data = dataset.train.data.astype(np.float64)
    test_data = dataset.test.data.astype(np.float64)
    std_train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
    std_test_data = (test_data - np.mean(test_data, axis=0)) / np.std(test_data, axis=0)
    train_data, val_data, train_labels, val_labels = train_test_split(std_train_data, dataset.train.labels, test_size=0.2, random_state=42)
    #model = import_model()
    model = lstm.Model()
    model.train(train_data, train_labels, hyperparams = {"validation": (val_data, val_labels),"epochs": 20, "batch_size": 32})
    predict = model.test(std_test_data, dataset.test.labels)
    result.append(predict)

    """
    for train_idx, val_idx in kf.split(dataset.train.data):
        train_data, val_data = dataset.train.data[train_idx], dataset.train.data[val_idx]
        train_labels, val_labels = dataset.train.labels[train_idx], dataset.train.labels[val_idx]
        model = import_model()
        model.train(train_data, train_labels, hyperparams = {"validation": (val_data, val_labels)})
        predict = model.test(dataset.test.data, dataset.test.labels)
        result.append(predict)
        print(predict)
    """

    print("confusion matrix: ")
    for predict in result:
        print(predict)
    return

if __name__ == "__main__":
    main()
    #main_cv()

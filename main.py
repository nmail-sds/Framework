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

from sklearn.model_selection import KFold 

import sys
sys.path.append("./data")


parser = argparse.ArgumentParser(description = "모델, 데이터 및 실행 모드 선택")

parser.add_argument("--model", type=str, required=True, help="select mode (e.g. linear_regression, dense, sgd_classifier)")
parser.add_argument("--dataset", type=str, required=True, help="select dataset (e.g. uci-secom, wafer)")
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
    
    '''
    dataset = import_dataset()
    kf = KFold(n_splits = n_splits)
    
    result = []

    for train_idx, val_idx in kf.split(dataset.train.data):
        train_data, val_data = dataset.train.data[train_idx], dataset.train.data[val_idx]
        train_labels, val_labels = dataset.train.labels[train_idx], dataset.train.labels[val_idx]
        model = import_model()
        model.train(train_data, train_labels, hyperparams = {"validation": (val_data, val_labels)})
        predict = model.test(dataset.test.data, dataset.test.labels)
        result.append(predict)
        print(predict)

    print("confusion matrix: ")
    for predict in result:
        print(predict)
    return

if __name__ == "__main__":
    #main()
    main_cv()

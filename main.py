'''
main.py
2018. 11. 22
Yihan Kim

사용법 : main.py --model [model_name] --data [data_name]

# --mode [train | test | all]
'''

import argparse
import os 
import importlib
import numpy as np
parser = argparse.ArgumentParser(description = "모델, 데이터 및 실행 모드 선택")

parser.add_argument("--model", type=str, help="select mode (e.g. linear_regression)")
parser.add_argument("--dataset", type=str, help="select dataset (e.g. uci-secom, wafer)")
#parser.add_argument("--mode", type=str, help="select mode (train, test or all)")

args = parser.parse_args()
# args.model, args.dataset, args.mode을 통해 입력값을 사용할 수 있음

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
        return read_data.main(args.dataset)
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

if __name__ == "__main__":
    main()

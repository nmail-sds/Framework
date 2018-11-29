'''
main.py
2018. 11. 22
Yihan Kim

사용법 : main.py --model [model name] --data [data name] --mode [train | test | all]
'''

import argparse
import os 
import importlib
import numpy as np
parser = argparse.ArgumentParser(description = "모델, 데이터 및 실행 모드 선택")

parser.add_argument("--model", type=str, help="select mode (e.g. linear_regression)")
parser.add_argument("--dataset", type=str, help="select dataset (e.g. uci-secom)")
parser.add_argument("--mode", type=str, help="select mode (train, test or all)")

args = parser.parse_args()
# args.model, args.dataset, args.mode을 통해 입력값을 사용할 수 있음

# args.model을 불러오기

from model import *
def import_model():
    # args.model에 해당하는 object 리턴
    # model = [args.model].Model()
    model = eval("{}.Model(debug=True)".format(args.model))
    #model = linear_regression.Model(debug=True)
    return model

from data import main as read_data
def import_dataset():
    try:
        dataset = read_data.main(os.path.join('./data', args.dataset))
        data = dataset[0, 1:, 1:-1].astype(np.float)
        labels = dataset[0, 1:, -1].astype(np.float) 
        return data, labels
    except:
        raise

def main():
    model = import_model()
    data, labels = import_dataset()
    model.train(data, labels)
    predict = model.test(data)
    label_ = (predict > 0).astype(np.int32) - (predict < 0).astype(np.int32)
    print(np.count_nonzero(labels - label_ != 0) / np.count_nonzero(labels > 0))
    print(np.count_nonzero(labels - label_ != 0) / np.count_nonzero(labels < 0))
    return    

if __name__ == "__main__":
    main()

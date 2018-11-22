'''
data/read.py
2018. 11. 22

데이터셋 이름을 입력받아 데이터를 불러오는 기능

'''

import os
import csv
import importlib
import numpy as np

def _read_csv(filename, basedir="."):
    filedir = os.path.join(basedir, filename)
    with open(filedir, 'r') as csv_f:
        data = list(csv.reader(csv_f))
    for i, row in enumerate(data):
        for j, item in enumerate(row):
            data[i][j] = '0.0' if item == '' else item
    return np.asarray(data)

def main(dataset):
    # dataset 이름 디렉토리의 데이터셋을 읽기
    data = []
    for filename in os.listdir(dataset):
        data.append(_read_csv(filename, basedir = dataset))
    return np.array(data)

if __name__ == "__main__":
    print(main('uci-secom').shape)


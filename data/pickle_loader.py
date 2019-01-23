import pickle as pkl
import numpy as np

dir = "C:/Users/djwls/PycharmProjects/Samsung_deep/Framework/data/"

def np2pkl(file_name, data):
    file_path = dir + file_name
    file_object = open(file_path, 'wb')
    pkl.dump(data, file_object)
    file_object.close()
    return None

def pkl2np(file_name):
    file_path = dir + file_name
    print(file_path)
    file_object = open(file_path, 'rb')
    data = pkl.load(file_object)
    return data
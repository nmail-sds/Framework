import pickle as pkl
import numpy as np
import os

dir = os.path.dirname(os.path.realpath(__file__))

def np2pkl(file_name, data):
    file_path = os.path.join(dir, file_name)
    file_object = open(file_path, 'wb')
    pkl.dump(data, file_object)
    file_object.close()
    return None

def pkl2np(file_name):
    file_path = os.path.join(dir, file_name)
    print(file_path)
    file_object = open(file_path, 'rb')
    data = pkl.load(file_object)
    return data

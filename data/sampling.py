'''
2019. 1. 21
SMOTE data sampling
Yihan Kim

usage : import sampling.smote as smote 

'''

from data.ds import Pair, Data
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

def smote_dataset(dataset: Data):
    # unpack 
    X = dataset.train.data 
    y = dataset.train.labels
    sm = SMOTE(random_state=42)
    X_res, Y_res = sm.fit_resample(X, y)
    return Data(X_res, Y_res, dataset.test.data, dataset.test.labels)

def main():
    X, y = make_classification(n_classes=2, class_sep=2,
             weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
             n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    
    p = smote_dataset(Data(X, y, None, None))
    X_ = p.train.data
    y_ = p.train.labels
    print(X_.shape)
    print(y_.shape)

if __name__ == "__main__":
    main()

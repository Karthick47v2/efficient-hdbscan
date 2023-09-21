import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import cProfile
import hdbscan

from lsh import LSHash

from time import time

if __name__ == '__main__':
    X, y = datasets.make_blobs(n_samples=10000, n_features=100, centers=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    # s = time()
    # clf = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', n_jobs=12)
    # clf.fit(X_train, y_train)
    # distances, indices = clf.kneighbors(X_train)
    # # print(X_train[indices[:,4]])
    # kth_distances = distances[:, 4]
    # # print(kth_distances[:10])

    # t = time()

    # # # # acc = np.sum(predictions == y_test) / len(y_test)
    # # # # print(kth_distances)
    # print('sklearn', t-s)

    # s = time()
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=5)#, min_samples=5)
    # clusterer.fit(X_train)
    # d = clusterer.labels_
    # t = time()

    # print('hdbscan', t-s)

    s = time()
    # lsh = LSHash(500, len(X_train[0]), 3)
    lsh = LSHash(20, len(X_train[0]), 5)
    lsh.add_points(X_train)
    # dist = lsh.nn_dist(20)


    def func():
        # lsh = LSHash(1000, len(X_train[0]), 10)
        # lsh.add_points(X_train)
        dist = lsh.nn_dist(5)
        
        
    
        
    # lsh.index_all(X_train)
    # dist = lsh.query(5)

    # print(lsh.precomputed_hashes)
    # print(lsh.hash_tables)

    cProfile.run('func()')



    t = time()

    # acc = np.sum(kth_distances == dist) / len(X_train)
    # print(kth_distances == dist)
    # print(acc)
    print('custom', t-s)

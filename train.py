import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import cProfile
import hdbscan

from lsh import LSHash
import math
from time import time

if __name__ == '__main__':
    X, y = datasets.make_blobs(
        n_samples=1000, n_features=100, centers=10, cluster_std=2.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.01, random_state=42)

    K = 3

    s = time()
    clf = KNeighborsClassifier(n_neighbors=K, algorithm='brute', n_jobs=12)
    clf.fit(X_train, y_train)
    distances, indices = clf.kneighbors(X_train)
    br_distances = distances[:, -1]
    # print(X_train)
    print()
    # print(X_train[indices[:,K-1]])
    # print(br_distances)
    t = time()
    print('sklearn-knn-bruteforce', t-s)

    # s = time()
    # clf = KNeighborsClassifier(n_neighbors=K, algorithm='kd_tree', n_jobs=12)
    # clf.fit(X_train, y_train)
    # distances, indices = clf.kneighbors(X_train)
    # kd_distances = distances[:, -1]
    # # print(kth_distances)
    # t = time()

    # print('sklearn-knn-kdtree', t-s)

    # s = time()
    # clf = KNeighborsClassifier(n_neighbors=K, algorithm='ball_tree', n_jobs=12)
    # clf.fit(X_train, y_train)
    # distances, indices = clf.kneighbors(X_train)
    # ball_distances = distances[:, -1]
    # # print(kth_distances)
    # t = time()

    # print('sklearn-knn-balltree', t-s)

    # # s = time()
    # # clusterer = hdbscan.HDBSCAN(min_cluster_size=5)#, min_samples=K)
    # # clusterer.fit(X_train)
    # # d = clusterer.labels_
    # # t = time()

    # # print('hdbscan', t-s)

    s = time()
    h = 1 * math.floor(math.log2(len(X_train) / K))
    n_t = math.ceil(math.log10(len(X_train[1])))
    n_t = 5 * h + 1   # 2 -> 10-10

    # n_t = 100

    if h == 0:
        h = 1
    # if n_t == 1:
    #     n_t = 10

    print('hyperplanes', h)
    print('# tables', n_t)

    lsh = LSHash(h, len(X_train[0]), n_t)  # h*3
    # lsh = LSHash(1, len(X_train[0]), 10)
    lsh.add_points(X_train)
    dist = lsh.calc_core_dist(K)
    t = time()

    print('lsh', t-s)

    # print(dist)

    # lsh = LSHash(h, len(X_train[0]), n_t)
    # lsh.add_points(X_train)
    # dist = lsh.calc_core_dist(K)

    # def func():
    #     dist = lsh.calc_core_dist(K)

    # cProfile.run('func()')

    # print('lsh', np.mean((np.abs(br_distances - dist)/br_distances) * 100))
    # print('sk-kd', np.mean((np.abs(br_distances - kd_distances)/br_distances) * 100))
    # print('sk-ball', np.mean((np.abs(br_distances - ball_distances)/br_distances) * 100))

    lsh_close_mask = np.isclose(dist, br_distances, atol=1e-9)
    # kd_close_mask = np.isclose(kd_distances, br_distances, atol=1e-9)
    # ball_close_mask = np.isclose(ball_distances, br_distances, atol=1e-9)

    # # print(br_distances == dist)
    # print('kd', np.sum(kd_close_mask) / len(X_train))
    # print('ball', np.sum(ball_close_mask) / len(X_train))
    print('lsh', np.sum(lsh_close_mask) / len(X_train))

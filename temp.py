import os
import ctypes
import argparse
import hdbscan

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from time import time

hdb_c_extension = ctypes.CDLL('./hdb.so')
graph_extension = ctypes.CDLL('./graph.so')

graph_extension.calc_mst.argtypes = [
    ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
graph_extension.calc_mst.restype = ctypes.POINTER(ctypes.c_int)

hdb_c_extension.calc_mutual_reachability_dist.argtypes = [ctypes.POINTER(
    ctypes.c_double), ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int]
hdb_c_extension.calc_mutual_reachability_dist.restype = ctypes.POINTER(
    ctypes.c_double)

parser = argparse.ArgumentParser(description='HDBSCAN')

parser.add_argument('--data', default='mnist', type=str, help='dataset')
parser.add_argument('--n', default=10000, type=int, help='number of rows')
parser.add_argument('--dim', default=100, type=int, help='number of cols')
parser.add_argument('--k_centers', default=5, type=int,
                    help='number of centriods')
parser.add_argument('--cluster_std', default=2.5,
                    type=float, help='cluster std')
parser.add_argument('--random_state', default=42,
                    type=int, help='random state')
parser.add_argument('--min_samples', default=5, type=int, help='min_samples')
parser.add_argument('--threads', default=48, type=int,
                    help='number of cpu threads')

args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.threads)

if args.data == 'online_retail':
    df = pd.read_excel('online_retail_II.xlsx')

    df = df[:10000]

    columns_to_remove = ['InvoiceDate', 'Invoice']
    df.drop(columns=columns_to_remove, axis=1, inplace=True)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    desc_vectorizer = TfidfVectorizer()
    text_data_transformed = desc_vectorizer.fit_transform(df['Description'])
    desc_tf_idf = pd.DataFrame(data=text_data_transformed.toarray(
    ), columns=desc_vectorizer.get_feature_names_out())

    price_scaler = StandardScaler()
    price_scaled = price_scaler.fit_transform(
        df['Price'].values.reshape(-1, 1))

    df['StockCode'] = df['StockCode'].astype(str)

    categorical_columns = ['StockCode', 'Customer ID', 'Country']
    label_encoder = LabelEncoder()
    df[categorical_columns] = df[categorical_columns].apply(
        label_encoder.fit_transform)

    X = np.concatenate((df[categorical_columns].values,
                       price_scaled, desc_tf_idf.values), axis=1)
elif args.data == 'bow':
    pass
elif args.data == 'iris':
    X, y = datasets.load_iris(return_X_y=True)
elif args.data == 'b_cancer':
    X, y = datasets.load_breast_cancer(return_X_y=True)
elif args.data == 'mnist':
    X, y = datasets.load_digits(return_X_y=True)
elif args.data == 'syn':
    X, y = datasets.make_blobs(
        n_samples=args.n, n_features=args.dim, centers=args.k_centers, cluster_std=args.cluster_std, random_state=args.random_state)
else:
    raise NotImplementedError("This dataset is not included.")

print('INSTANCES:', X.shape[0], ' FEATURES: ', X.shape[1])

data = X
k_nearest_neighbors_value = args.min_samples


sh = time()
clusterer = hdbscan.HDBSCAN(min_cluster_size=k_nearest_neighbors_value,
                            min_samples=k_nearest_neighbors_value, match_reference_implementation=False)
clusterer.fit(data)
print('HDB', time() - sh)

st = time()
n, dim = data.shape
data_flat = data.flatten().astype(ctypes.c_double)

mutual_reachability_dist = hdb_c_extension.calc_mutual_reachability_dist(
    data_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n, dim, k_nearest_neighbors_value)
c_cluster_labels = graph_extension.calc_mst(n, mutual_reachability_dist,
                                            k_nearest_neighbors_value+1)

print('Time taken', time() - st)

c_cluster_labels = [c_cluster_labels[i] for i in range(n)]
sk_hdb = Counter(clusterer.labels_)
my_hdb = Counter(c_cluster_labels)


new_sk_hdb = sorted({key: value for key, value in sk_hdb.items()
                    if key != -1}.values(), reverse=True)
new_my_hdb = sorted({key: value for key, value in my_hdb.items()
                    if key != -1}.values(), reverse=True)


sk_metric = 0
my_metric = 0

if args.data in ['mnist', 'iris', 'b_cancer']:
    orig_labels = Counter(y)
    new_orig = sorted(orig_labels.values(), reverse=True)
    ttl = sum(new_orig)
    for orig, sk, my in zip(new_orig, new_sk_hdb, new_my_hdb):
        sk_metric += abs(orig - sk)
        my_metric += abs(orig - my)
    print('ARI', sk_metric/ttl, my_metric/ttl)
print(sum(new_sk_hdb)/len(X), sum(new_my_hdb)/len(X))


if args.dim == 2:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=50)
    axs[0].set_title('Original')

    axs[1].scatter(X[:, 0], X[:, 1], c=clusterer.labels_,
                   cmap='plasma', edgecolors='k', s=50)
    axs[1].set_title('SK Learn')

    axs[2].scatter(X[:, 0], X[:, 1], c=c_cluster_labels,
                   cmap='magma', edgecolors='k', s=50)
    axs[2].set_title('Ours')

    plt.tight_layout()
    plt.savefig('res.png')

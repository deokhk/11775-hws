#!/bin/python

import argparse 
import pickle
import sys
import time

import pandas as pd
from sklearn.cluster.k_means_ import KMeans


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str,
                        help='Path to the .mfcc.csv file with sub-sampled features')

    parser.add_argument('-k', type=int,
                        help='Number of clusters')

    parser.add_argument('-o', '--output_path', type=str,
                        help='Path to the file where the model K-Means model will be stored')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # 1. load all mfcc features in one array
    selection = pd.read_csv(args.input_path, sep=';', dtype='float')
    X = selection.values
    start = time.time()
    kmeans = KMeans(n_clusters=args.k, random_state=0, n_jobs=10).fit(X)
    end = time.time()

    # 2. Save trained model
    pickle.dump(kmeans, open(args.output_path, 'wb'))

    print(f'K-means model with {args.k} centroid trained successfully!')
    print(f'Model saved to:            {args.output_path}')
    print(f'Time elapsed for training: {end-start}')

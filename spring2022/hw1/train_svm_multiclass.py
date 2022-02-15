#!/bin/python

import numpy as np
import os
from sklearn.svm import SVC
import pickle
import argparse
import wandb
import sys
import pdb
from sklearn.model_selection import train_test_split

# Train SVM

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':
  args = parser.parse_args()
  # 0. Set up wandb
  wandb.init(project="MultAnalysis_HW1", entity="deokhk")
  wandb.config.update(args)

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignore
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

      label_list.append(int(df_videos_label[video_id]))

  print("number of samples: %s" % len(feat_list))
  y = np.array(label_list)
  X = np.array(feat_list)
  print(f'X: {X.shape}')
  print(f'y: {y.shape}')
  
  # Split training and validation examples
  train_size = 0.9
  X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = train_size)

  print(f'Train data: {X_train.shape}, Validation data: {X_val.shape}')

  # pass array for svm training
  # one-versus-rest multiclass strategy
  clf = SVC(cache_size=2000, decision_function_shape='ovr', kernel="rbf")
  clf.fit(X, y)

  # save trained SVM in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  print('One-versus-rest multi-class SVM trained successfully')

  print('Now evaluating the MLP classifiers..')
  accuracy = clf.score(X_val, y_val)
  wandb.log({"Val_accuracy": accuracy})
  print(f"Validation accuracy: {accuracy}")
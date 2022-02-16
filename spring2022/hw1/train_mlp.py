#!/bin/python

import argparse
import os
import pickle
import wandb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

import sys

# Train MLP classifier with labels

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
    # for videos with no audio, ignored in training
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

      label_list.append(int(df_videos_label[video_id]))


  print("number of samples: %s" % len(feat_list))
  X = np.array(feat_list)
  y = np.array(label_list)

  print(f'X: {X.shape}')
  print(f'y: {y.shape}')
  
  # Split training and validation examples
  train_size = 0.9
  X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = train_size)

  print(f'Train data: {X_train.shape}, Validation data: {X_val.shape}')


  clf = MLPClassifier(hidden_layer_sizes=(1024, 512),
                      activation="relu",
                      solver="adam",
                      alpha=1e-3)
  clf.fit(X_train, y_train)

  # save trained MLP in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  print('MLP classifier trained successfully')

  print('Now evaluating the MLP classifiers..')
  accuracy = clf.score(X_val, y_val)
  wandb.log({"Val_accuracy": accuracy})
  print(f"Validation accuracy: {accuracy}")

  print("Now generating confusion matrix..")
  """
  label = ['blowing_out_candles',
    'dribbling_basketball',
    'flipping_pancake',
    'getting_a_haircut',
    'hitting_baseball',
    'home_roasting_coffee',
    'motorcycling',
    'mowing_lawn',
    'playing_drums',
    'playing_guitar',
    'playing_piano',
    'shoveling_snow',
    'singing',
    'tapping_pen',
    'tickling'
  ]
  """
  # First, plot non-normalized version.
  plot = plot_confusion_matrix(clf,
                              X_val, y_val,
                              cmap=plt.cm.Blues,
                              normalize=None)
  plot.ax_.set_title('Confusion Matrix')
  plot.figure_.savefig('Non-normalized.png')
  # Second, plot normalized version.
  plot_normalized = plot_confusion_matrix(clf,
                              X_val, y_val,
                              cmap=plt.cm.Greens,
                              normalize='true')
  plot_normalized.ax_.set_title('Confusion Matrix')
  plot_normalized.figure_.savefig('Normalized.png')

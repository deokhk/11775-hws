#!/bin/python

import argparse
from operator import not_
import os
import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier

# Apply the MLP model to the testing videos;
# Output prediction class for each video

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_file")
  parser.add_argument("feat_dir")
  parser.add_argument("feat_dim", type=int)
  parser.add_argument("list_videos")
  parser.add_argument("output_file")
  parser.add_argument("--file_ext", default=".csv")

  return parser.parse_args()


if __name__ == '__main__':

  args = parse_args()

  # 1. load mlp model
  mlp = pickle.load(open(args.model_file, "rb"))

  # 2. Create array containing features of each sample
  fread = open(args.list_videos, "r")
  feat_list = []
  video_ids = []
  not_found_count = 0
  for line in fread.readlines()[1: ]:
    video_id = line.split(',')[0].strip()
    video_ids.append(video_id)
    feat_filepath = os.path.join(args.feat_dir, video_id + args.file_ext)
    if not os.path.exists(feat_filepath):
      feat_list.append(np.zeros(args.feat_dim))
      not_found_count += 1
    else:
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

  if not_found_count > 0:
    print(f'Could not find the features for {not_found_count} samples.')

  X = np.array(feat_list)
  print(f'X: {X.shape}')

  # 3. Get predictions
  # (num_samples) with integer
  pred_classes = mlp.predict(X)

  # 4. save for submission
  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, pred_class in enumerate(pred_classes):
      f.writelines("%s,%d\n" % (video_ids[i], pred_class))

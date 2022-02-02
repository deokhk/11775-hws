#!/bin/python
# Randomly select MFCC frames

import argparse
import os

import numpy
from tqdm import tqdm


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_path", type=str,
                      help='Path to the txt file with list of file names and labels')

  parser.add_argument("--ratio", type=float, default=0.2,
                      help='Proportion of data that will be randomly sampled')

  parser.add_argument("--mfcc_dir", type=str, default="./mfcc",
                      help='')

  parser.add_argument("--output_path", type=str,
                      help='Path to the file where the seleted MFCC samples will be stored')

  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()

  fread = open(args.input_path, "r")
  fwrite = open(args.output_path, "w")

  # random selection is done by randomizing the rows of the whole matrix, and then selecting the first
  # num_of_frame * ratio rows
  numpy.random.seed(18877)

  for line in tqdm(fread.readlines()[1:]):  # skipped the header
    mfcc_dir = os.path.join(args.mfcc_dir, line.strip().split(",")[0] + ".mfcc.csv")
    if not os.path.exists(mfcc_dir):
      continue
    array = numpy.genfromtxt(mfcc_dir, delimiter=";")
    numpy.random.shuffle(array)
    select_size = int(array.shape[0] * args.ratio)
    feat_dim = array.shape[1]

    for n in range(select_size):
      line = str(array[n][0])
      for m in range(1, feat_dim):
        line += ";" + str(array[n][m])
      fwrite.write(line + "\n")
  fwrite.close()

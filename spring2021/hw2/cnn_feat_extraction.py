import argparse
import cv2
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("video_path")
parser.add_argument("output_path")
parser.add_argument("--feat_fps", type=float, default=1.5,
                    help="frame rate for getting features")
parser.add_argument("--model", default="resnet18",
                    help="pytorch ImageNet model to use")
parser.add_argument("--feat_layer", default="avgpool",
                    help="which layer to extract")
parser.add_argument("--feat_size", type=int, default=512,
                    help="the extracted layer's dimension")
parser.add_argument("--use_gpu", action="store_true")



class Get_CNN():
  def __init__(self, cuda=False, model_name='resnet18', layer='avgpool', layer_output_size=512):
    """ Img2Vec
    :param cuda: If set to True, will run forward pass on GPU
    :param model_name: String name of requested model
    :param layer and layer_output_size: layer and its output size
    """
    # Model
    raise Exception("Please implement Get_CNN()")

    # Transforms


  def get_emb(self, img):
    """ Get vector embedding from PIL image
    :param img: PIL Image
    :returns: Numpy ndarray of size (layer_output_size,)
    """
    raise Exception("Please implement get_emb()")



def get_cnn_features_from_video(cnn_model,
                                video_filepath,
                                cnn_feat_filepath,
                                keyframe_interval):
  " Extracts features in the given keyframe_interval. "
  " Saves features in csv file."

  cnn_feats = []

  for keyframe in get_keyframes(video_filepath, keyframe_interval):
    # (layer_output_size,)
    emb = model.get_emb(Image.fromarray(cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB)))
    cnn_feats.append(emb)
  if cnn_feats:
    # global average pooling
    cnn_feat = np.mean(cnn_feats, axis=0)
    np.savetxt(cnn_feat_filepath, cnn_feat)
  else:
    tqdm.write("warning, %s has empty features." % os.path.basename(video_filepath))

def get_keyframes(video_filepath, keyframe_interval):
  """Generator function which returns the next keyframe.
  Args:
      video_filepath (string): path to the video
      keyframe_interval (int): return a frame every k frame
  Returns:
      frame (np.array): opencv loaded RGB frame object
  """

  video_cap = cv2.VideoCapture(video_filepath)
  # TODO: implement your code here
  # it should be a generator that yields a frame when it should
  raise Exception("Please implement get_keyframes")
  video_cap.release()

if __name__ == "__main__":
  args = parser.parse_args()

  model = Get_CNN(
      cuda=args.use_gpu,
      model_name=args.model,
      layer=args.feat_layer, layer_output_size=args.feat_size)

  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

  video_fps = 30.0  # assuming the video frame rate
  # extract feature from every k frame
  frame_interval = int(video_fps / args.feat_fps)

  # Loop over all videos (training, val, testing)
  video_files = glob(os.path.join(args.video_path, "*.mp4"))

  for video_file in tqdm(video_files):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    cnn_feat_outfile = os.path.join(args.output_path, video_name + ".csv")
    if os.path.exists(cnn_feat_outfile):
      continue

    # Get SURF features for one video
    get_cnn_features_from_video(model, video_file,
                                cnn_feat_outfile, frame_interval)



# Instructions for using the pipelines

Here is a detailed step-by-step guide for using the media event detection(MED) pipeline.

## Download the data

Please download the data from [this link](https://drive.google.com/file/d/1WEINPdvQ1ZUELxaXlhHcvoOjEML8gYYY/view?usp=sharing). 
Please download the data under (11775-hws/spring2022/hw1/). 
Unzip the downloaded data by
```
$ unzip 11775_s22_data.zip
```
After you unzip the data, there will be two folders(videos/, labels/) inside the 11755_s22_data. Move them under (11775-hws/spring2022/hw1/).


## Step-by-step guidelines for the MED pipeline
#### 

Suppose you are at (11775-hws/spring2022/hw1). Now I'll refer this path as `hw1`.
Create  folders we need.
```
$ mkdir mp3
```

Install FFMPEG and other dependencies by:
```
$ sudo pip install sklearn pandas tqdm
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
$ conda install -c conda-forge ffmpeg
```
Clone  [this github repo](https://github.com/salmedina/soundnet_pytorch) to the `hw1`.

### Extract SoundNet Features

First, you need to extract audio from the given videos. Specifically, we will extract audio with mp3 format and 22050 Hz sampling rate and 1 channels.
```
$ for file in videos/*;do filename=$(basename $file .mp4); ffmpeg -y -i $file -ac 1 -f mp3 -ar 22050 mp3/${filename}.mp3; done
```
Then, extract features from the soundNet model. First Move to the cloned repository and make some folders we need. This will take about an hour from the instance.
```
$ cd soundnet_pytorch
$ mkdir soundnet
$ mkdir soundnet/raw
$ python3 -u extract_feats.py -m models/sound8.npy -i ./mp3 -o ./soundnet/raw -f .mp3
```
Now, we will perform avg pooling to get single vector representation from a video.
```
$ mkdir soundnet/avg_pooling
$ mkdir soundnet/avg_pooling/y_scns
$ python3 -u get_avg_pool.py -i ./soundnet/raw -o ./soundnet/avg_pooling/y_scns -f y_scns
```


### Train the MLP

Suppose you are under `hw1` directory. 
Train MLP by:
```
$ python train_mlp.py soundnet_pytorch/soundnet/avg_pooling/y_scns/ 1024 labels/trainval.csv models/soundnet-1024.mlp.model
```
This will give about 51~52% classification accuracy on the validation set.
Test:
```
$ mkdir results
$ python test_mlp.py models/soundnet-1024.mlp.model soundnet_pytorch/soundnet/avg_pooling/y_scns/ 1024 labels/test_for_students.csv results/soundnet-1024.mlp.csv
```






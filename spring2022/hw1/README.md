# Instructions for hw1

In this homework we will perform a video classification task using audio-only features.

## Data and Labels

Please download the data from [this link](https://drive.google.com/file/d/1WEINPdvQ1ZUELxaXlhHcvoOjEML8gYYY/view?usp=sharing). The `.zip` file should include the following:
1. `video/` folder with 8249 videos in MP4 format
2. `labels/` folder with two files:
    - `cls_map.csv`: csv with the mapping of the labels and its corresponding class ID (*Category*)
    - `train_val.csv`: csv with the Id of the video and its label
    - `test_for_students.csv`: submission template with the list of test samples


## Step-by-step baseline instructions

For the baselines, we will provide code and instructions for two feature representations (MFCC-Bag-of-Features and [SoundNet-Global-Pool](https://arxiv.org/pdf/1610.09001.pdf)) and two classifiers (SVM and MLP). Assuming you are under Ubuntu 18.04 system and under this directory (11775-hws/spring2022/hw1/).

First uncompress the data into this folder simply by running:
```
$ unzip 11775_s22_data.zip
```

### MFCC-Bag-Of-Features

Let's create the folders we need first:
```
$ mkdir wav/ mfcc/ bof/
```

1. Dependencies: FFMPEG, OpenSMILE, Python: sklearn, pandas

Download OpenSMILE 3.0 from [here](https://github.com/audeering/opensmile/releases/download/v3.0.0/opensmile-3.0-linux-x64.tar.gz) and extract under the `./tools/` directory:
```
$ tar -zxvf opensmile-3.0-linux-x64.tar.gz
```

Install FFMPEG by:
```
$ sudo apt install ffmpeg
```

Install python dependencies by:
```
$ sudo pip install sklearn pandas tqdm
```

2. Get MFCCs

We will first extract the audio from the videos:

```
$ for file in videos/*;do filename=$(basename $file .mp4); ffmpeg -y -i $file -ac 1 -f wav wav/${filename}.wav; done
```
Then run OpenSMILE to get MFCCs into CSV files. We will directly run the binaries of OpenSMILE (no need to install):

```
$ for file in wav/*;do filename=$(basename $file .wav); ./tools/opensmile-3.0.1-linux-x64/bin/SMILExtract -C config/MFCC12_0_D_A.conf -I ${file} -O mfcc/${filename}.mfcc.csv;done
```

The above should take 1-2 hours. If you generate less `.wav` or `.mfcc` files that video, do not worry, that is normal in real-life scenarios. Report which video files had trouble and investigate the reason of the failure.

3. K-Means clustering

As taught in the class, we will use K-Means to get feature codebook from the MFCCs. Since there are too many feature lines, we will randomly select a subset (20%) for K-Means clustering by:
```
$ python select_frames.py --input_path labels/train_val.csv --ratio 0.2 --output_path mfcc/selected.mfcc.csv --mfcc_dir mfcc/
```

Now we train it by (50 clusters, this would take about 7-15 minutes):
```
$ python train_kmeans.py -i selected.mfcc.csv -k 50 -o models/kmeans.50.model
```

4. Feature extraction

Now we have the codebook, we will get bag-of-features (a.k.a. bag-of-words) using the codebook and the MFCCs. First, we need to get video names:
```
$ ls videos/ | while read line;do filename=$(basename $line .mp4);echo $filename;done > videos.name.lst
```


Now we extract the feature representations for each video:
```
$ python get_bof.py kmeans.50.model 50 videos.name.lst --mfcc_path mfcc/ --output_path bof/
```

Now you can follow [here](#svm-classifier) to train SVM classifiers or [MLP](#mlp-classifier) ones.


### SoundNet-Global-Pool

Just as the MFCC-Bag-Of-Feature, we could also use the [SoundNet](https://arxiv.org/pdf/1610.09001.pdf) model to extract a vector feature representation for each video. Since SoundNet is trained on a large dataset, this feature is usually better compared to MFCCs.

Please follow [this Github repo](https://github.com/salmedina/soundnet_pytorch) to extract audio features. Please read the paper and think about what layer(s) to use. If you save the feature representations in the same format as in the `bof/` folder, you can directly train SVM and MLP using the following instructions.


### SVM classifier

From the previous sections, we have extracted two fixed-length vector feature representations for each video. We will use them separately to train classifiers.

Suppose you are under `hw1` directory. Train SVM by:
```
$ mkdir models/
$ python train_svm_multiclass.py bof/ 50 labels/trainval.csv models/mfcc-50.svm.multiclass.model
```

Run SVM on the test set:
```
$ python test_svm_multiclass.py models/mfcc-50.svm.multiclass.model bof/ 50 labels/test_for_student.label mfcc-50.svm.multiclass.csv
```


### MLP classifier

Suppose you are under `hw1` directory. Train MLP by:
```
$ python train_mlp.py bof/ 50 labels/trainval.csv models/mfcc-50.mlp.model
```

Test:
```
$ mkdir results
$ python test_mlp.py models/mfcc-50.mlp.model bof 50 labels/test_for_student.label results/mfcc-50.mlp.csv
```


### Submission to Kaggle

You can then submit the test outputs to the leaderboard:
```
https://www.kaggle.com/c/11775-s22-hw1
```
We use classification accuracy as the evaluation metric. Please refer to in the data to the `test_for_students.csv` file for submission format.


### Things to try to improve your model performance

Now here comes the fun part. You can start experimenting with the code and exploring how you can improve your model performance. Some hints:

+ Split `trainval.csv` into `train.csv` and `val.csv` to validate your model variants. This is important since the leaderboard limits the number of times you can submit, which means you cannot test most of your experiments on the official test set.
+ Try different number of K-Means clusters
+ Try different layers of SoundNet
+ Try out other audio features such as [VGGSound](https://github.com/hche11/VGGSound) or the latest model for audio classification [PaSST](https://github.com/kkoutini/passt).
+ Try different classifiers (different SVM kernels, different MLP hidden sizes, etc.). Please refer to [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier) documentation.
+ Try different fusion or model aggregation methods. For example, you can simply average two model predictions (late fusion).

# Baselines
This is a place to keep track of all your baselines per release.

| Date     | Release | Loss   |
| -------- | ------- | ------ |
| 07/05/21 | 0.0.1   | 0.1534 |

# Long Description of Network
This is where you can added a longer description of your network.

# How to Organise This Module
There are 4 core files that should be populated in every repository. Not all the code needs to be contained within these files, but please set them up so that they can be used as entrypoints to your code base. (i.e another developer can see open these files up to see what methods and classes you have imported to run your models). However, for cases where you are creating multiple networks or using a lot of data functionality please feel free to use more files. There is some additional information below.

Note: Feel free to use any additional scripts e.g loss.py to store your loss methods, or constant.py to store any constant. But the following scripts are essential to be populated.

## data.py
This script should contain all classes and methods needed to pre-process, generate and augment your data.

If you are using multiple versions of data, feel free to use more data.py scripts, for instance `data_pretrain.py` if you wanted to group methods used for pretraining.

## inference.py
This script when run as a module should performance inference. No additional steps should be expected to be performed before running this script. Any preprocessing or data handling should be performed within the script. Argparse can be used to allow different arguments to be used.

To run this script do the following:
```bash
$ python -m template.inference
```

## model.py
This script should contain all classes and methods needed to define your network architecture

## train.py
This script when run as a module should performance training. No additional steps should be expected to be performed before running this script. Any preprocessing or data handling should be performed within the script. Argparse can be used to allow different arguments to be used.

To run this script do the following:
```bash
$ python -m template.train
```

Similarly, if you are training multiple networks, please define multiple train scripts e.g `train_A.py`, `train_B.py`.

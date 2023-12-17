import os
import sys
sys.path.append("../")
from models.DSSM import Net
from models.LSTMBasic import Net
from models.LSTMBid import Net
from models.LSTMBidAtten import Net
from models.LSTMMultiLayerBidAttn import Net
# 'number of epochs to train'
max_epochs = 200

start_epoch = 1

use_bert = True
pretrain_path = "/home/cuidongdong/ERNIE_1.0_max-len-512-pytorch"

datasets = "xx"

# number of workers to load training data
nw = 8
require_improvement = 1000
# path to save models
# output_dir = "./object_detection/YOLO_v2/output/"
save_path = "./checkpoints/"

# use tensorboard
use_tfboard = False

# print it every 10 times
display_interval = 50

# save models every 20 epoch
save_interval = 20

# use multi gpus
mGPUs = True

# use cuda
cuda = True

# train models from checkpoints
resume = False

checkpoint_epoch = 180

saturation = 1.5
exposure = 1.5
hue = .1

jitter = 0.3

thresh = .6

batch_size = 3

lr = 0.001

decay_lrs = {
    60: 0.00001,
    90: 0.000001
}

momentum = 0.9
weight_decay = 0.0005

# multi-scale training:
# {k: epoch, v: scale range}
multi_scale = True

# number of steps to change input size
scale_step = 40  # 每40次iter就开始改变输入尺寸的大小

scale_range = (0, 4)

epoch_scale = {
    1: (0, 8),
    15: (2, 5),
    30: (1, 6),
    60: (0, 7),
    75: (0, 9)
}

input_sizes = [(320, 320),
               (352, 352),
               (384, 384),
               (416, 416),
               (448, 448),
               (480, 480),
               (512, 512),
               (544, 544),
               (576, 576)]

input_size = (416, 416)

test_input_size = (416, 416)

strides = 32

debug = False

classes = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


object_scale = 5
noobject_scale = 1
class_scale = 1
coord_scale = 1

saturation = 1.5
exposure = 1.5
hue = .1

jitter = 0.3

thresh = .6

batch_size = 32

lr = 0.0001

decay_lrs = {
    60: 0.00001,
    90: 0.000001
}

momentum = 0.9
weight_decay = 0.0005


# multi-scale training:
# {k: epoch, v: scale range}
multi_scale = True

# number of steps to change input size
scale_step = 40  # 每40次iter就开始改变输入尺寸的大小

scale_range = (0, 4)

epoch_scale = {
    1:  (0, 8),
    15: (2, 5),
    30: (1, 6),
    60: (0, 7),
    75: (0, 9)
}

input_sizes = [(320, 320),
               (352, 352),
               (384, 384),
               (416, 416),
               (448, 448),
               (480, 480),
               (512, 512),
               (544, 544),
               (576, 576)]

input_size = (416, 416)

test_input_size = (416, 416)

strides = 32

debug = False

import time
timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(timestr)
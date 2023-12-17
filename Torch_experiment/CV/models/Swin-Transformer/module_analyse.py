import os
import time
import json
import random
import argparse
import datetime
import io
import os
import time
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from data.zipreader import is_zip_path, ZipReader

import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data as data

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
from models import build_model

from torchvision import datasets, transforms
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default="configs/swin/swin_base_patch4_window7_224.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default="/home/cuidongdong/data/flower_photos", help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


# def find_classes(dir):
#     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#     classes.sort()
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return classes, class_to_idx

#
# def has_file_allowed_extension(filename, extensions):
#     """Checks if a file is an allowed extension.
#     Args:
#         filename (string): path to a file
#     Returns:
#         bool: True if the filename ends with a known image extension
#     """
#     filename_lower = filename.lower()
#     return any(filename_lower.endswith(ext) for ext in extensions)
#
# try:
#     from torchvision.transforms import InterpolationMode
#
#
#     def _pil_interp(method):
#         if method == 'bicubic':
#             return InterpolationMode.BICUBIC
#         elif method == 'lanczos':
#             return InterpolationMode.LANCZOS
#         elif method == 'hamming':
#             return InterpolationMode.HAMMING
#         else:
#             # default bilinear, do we want to allow nearest?
#             return InterpolationMode.BILINEAR
#
#
#     import timm.data.transforms as timm_transforms
#
#     timm_transforms._pil_interp = _pil_interp
# except:
#     from timm.data.transforms import _pil_interp

# def make_dataset(dir, class_to_idx, extensions):
#     images = []
#     dir = os.path.expanduser(dir)
#     for target in sorted(os.listdir(dir)):
#         d = os.path.join(dir, target)
#         if not os.path.isdir(d):
#             continue
#
#         for root, _, fnames in sorted(os.walk(d)):
#             for fname in sorted(fnames):
#                 if has_file_allowed_extension(fname, extensions):
#                     path = os.path.join(root, fname)
#                     item = (path, class_to_idx[target])
#                     images.append(item)
#
#     return images


# def make_dataset_with_ann(ann_file, img_prefix, extensions):
#     images = []
#     with open(ann_file, "r") as f:
#         contents = f.readlines()
#         for line_str in contents:
#             path_contents = [c for c in line_str.split('\t')]
#             im_file_name = path_contents[0]
#             class_index = int(path_contents[1])
#
#             assert str.lower(os.path.splitext(im_file_name)[-1]) in extensions
#             item = (os.path.join(img_prefix, im_file_name), class_index)
#
#             images.append(item)
#
#     return images
#
# class DatasetFolder(data.Dataset):
#     """A generic data loader where the samples are arranged in this way: ::
#         root/class_x/xxx.ext
#         root/class_x/xxy.ext
#         root/class_x/xxz.ext
#         root/class_y/123.ext
#         root/class_y/nsdf3.ext
#         root/class_y/asd932_.ext
#     Args:
#         root (string): Root directory path.
#         loader (callable): A function to load a sample given its path.
#         extensions (list[string]): A list of allowed extensions.
#         transform (callable, optional): A function/transform that takes in
#             a sample and returns a transformed version.
#             E.g, ``transforms.RandomCrop`` for images.
#         target_transform (callable, optional): A function/transform that takes
#             in the target and transforms it.
#      Attributes:
#         samples (list): List of (sample path, class_index) tuples
#     """
#
#     def __init__(self, root, loader, extensions, ann_file='', img_prefix='', transform=None, target_transform=None,
#                  cache_mode="no"):
#         # image folder mode
#         if ann_file == '':
#             _, class_to_idx = find_classes(root)
#             samples = make_dataset(root, class_to_idx, extensions)
#         # zip mode
#         else:
#             samples = make_dataset_with_ann(os.path.join(root, ann_file),
#                                             os.path.join(root, img_prefix),
#                                             extensions)
#
#         if len(samples) == 0:
#             raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
#                                 "Supported extensions are: " + ",".join(extensions)))
#
#         self.root = root
#         self.loader = loader
#         self.extensions = extensions
#
#         self.samples = samples
#         self.labels = [y_1k for _, y_1k in samples]
#         self.classes = list(set(self.labels))
#
#         self.transform = transform
#         self.target_transform = target_transform
#
#         self.cache_mode = cache_mode
#         if self.cache_mode != "no":
#             self.init_cache()
#
#     def init_cache(self):
#         assert self.cache_mode in ["part", "full"]
#         n_sample = len(self.samples)
#         global_rank = dist.get_rank()
#         world_size = dist.get_world_size()
#
#         samples_bytes = [None for _ in range(n_sample)]
#         start_time = time.time()
#         for index in range(n_sample):
#             if index % (n_sample // 10) == 0:
#                 t = time.time() - start_time
#                 print(f'global_rank {dist.get_rank()} cached {index}/{n_sample} takes {t:.2f}s per block')
#                 start_time = time.time()
#             path, target = self.samples[index]
#             if self.cache_mode == "full":
#                 samples_bytes[index] = (ZipReader.read(path), target)
#             elif self.cache_mode == "part" and index % world_size == global_rank:
#                 samples_bytes[index] = (ZipReader.read(path), target)
#             else:
#                 samples_bytes[index] = (path, target)
#         self.samples = samples_bytes
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return sample, target
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         fmt_str += '    Root Location: {}\n'.format(self.root)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         tmp = '    Target Transforms (if any): '
#         fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        # 这里要修改一下 5
        nb_classes = 5
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if isinstance(path, bytes):
        img = Image.open(io.BytesIO(path))
    elif is_zip_path(path):
        data = ZipReader.read(path)
        img = Image.open(io.BytesIO(data))
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    return img.convert('RGB')


args, config = parse_option()
# 在这个操作之后才可以进行设置，否则不设置会报错，它和config.freeze()函数类似
config.defrost()
dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)

data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        # batch_size=config.DATA.BATCH_SIZE,
        batch_size=2,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

mixup_fn = None
mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
if mixup_active:  # mixup is a strategy to make data augment
    mixup_fn = Mixup(
        mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
        prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
        label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

count = 0
print(len(data_loader_train))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
# Creating model:swin/swin_base_patch4_window7_224
for data in data_loader_train:
    print(data[0].shape, data[1])
    count += 1
    samples, targets = mixup_fn(data[0].to(device), data[1].to(device))
    print('sample ', samples.shape)
    print('targets ', targets)

    model = build_model(config)
    model.to(device)
    output = model(samples)
    print('output: ', output.shape)
    if count >= 1:
        break


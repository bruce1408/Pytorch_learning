import sys
sys.path.append("../../..")

import argparse
import os
from datetime import datetime
from typing import Tuple
import torch
import mmcv
from torchvision import transforms
from torchvision.datasets.folder import default_loader
# from torchvision import models
# import json
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
# import image_net_config

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def resize_image(srcimg):
    inpWidth = 640
    inpHeight = 640

    keep_ratio = True
    padh, padw, newh, neww = 0, 0, inpHeight, inpWidth
    if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
        hw_scale = srcimg.shape[0] / srcimg.shape[1]
        if hw_scale > 1:
            newh, neww = inpHeight, int(inpWidth / hw_scale)
            img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
            padw = int((inpWidth - neww) * 0.5)
            img = cv2.copyMakeBorder(img, 0, 0, padw, inpWidth - neww - padw, cv2.BORDER_CONSTANT,
                                        value=0)  # add border
        else:
            newh, neww = int(inpHeight * hw_scale), inpWidth
            img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
            padh = int((inpHeight - newh) * 0.5)
            img = cv2.copyMakeBorder(img, padh,inpHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        img = cv2.resize(srcimg, (inpWidth, inpHeight), interpolation=cv2.INTER_AREA)
    return img, newh, neww, padh, padw


def normalize_img(img):  
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    return img


def get_val_transform():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    size = (640, 640)
    val_transforms = transforms.Compose([
        # transforms.Resize(size + 24),
        # transforms.Resize(size),
        # transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize]
                                        )
    return val_transforms


def transform_one_image(image_path):
    '''return 1x3x640x640 data'''
    img = default_loader(image_path)
    transform = get_val_transform()
    data = transform(img)
    data = torch.unsqueeze(data, 0)    
    return data



def parse_folder2class():
    folder2class = {}
    with open("imagenet_class.txt") as f:
        for each in f:
            line = each.strip()
            if line == "": continue
            strs = line.split()
            folder2class[strs[1]] = int(strs[0])
    return folder2class


def calibration_data_generate(img_path, save_path):

    #get dict from folder to class id
    #transform
    if not os.path.exists(save_path):
        print("the save path is not exist, mkdir the folders")
        os.makedirs(save_path)
    dataset_prefix="val_yolop_v1"
    root_dir = f"/root/bdd100k_images/{dataset_prefix}"

    count = 0
    dst_dir = f"/root/bdd100k_images/{dataset_prefix}_raw"
    if not os.path.exists(dst_dir): os.makedirs(dst_dir)
    
    img_paths = []
    for name in tqdm(os.listdir(root_dir)):
        img_path = os.path.join(root_dir, name)
        data = transform_one_image(img_path)
        dst_path = os.path.join(dst_dir, f"{dataset_prefix}_{Path(name).stem}.raw")
        data.numpy().tofile(dst_path)
        img_paths.append(("yolop_val_raw/"+Path(dst_path).name))
        count += 1
        if count >= 200: break   

    #save    
    with open("imagenet_raw_class.txt", "w") as f:
        for each in img_paths:
            f.write(f"{each}\n")
    

def calibration_yolop(img_dir):
    img_list = os.listdir(img_dir)
    count = 0
    
    for img in tqdm(img_list):
        if count > 200:
            break
        img_path = os.path.join(img_dir, img)
        # print(img_path)
        img_data = cv2.imread(img_path)
        img_value = resize_image(img_data)[0]
        save_img = os.path.join("/root/bdd100k_images/val_yolop_v1", img)
        cv2.imwrite(save_img, img_value)
        count += 1
        

def calibration_yolop_preprocess(img_dir):
    img_list = os.listdir(img_dir)
    count = 0
    for img_name in tqdm(img_list):
        if count > 200:
            break
        img_path = os.path.join(img_dir, img_name)
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        heigh, width, _ = img.shape
        print(heigh, width)
        r = min(640/heigh, 640/width)
        
        new_w = int(width * r)
        new_h = int(heigh * r)
        
        pad_w = 640 - new_w
        pad_h = 640 - new_h
        
        left = int(pad_w / 2)
        top = int(pad_h / 2)
        right = 640 - new_w - left
        bottom = 640 - new_h - top
        if ((heigh != new_h) or (width != new_w )):
            img = cv2.resize(img, (new_w, new_h), cv2.INTER_AREA)

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        cv2.imshow("after show", img)
        cv2.waitKey(0)
        cv2.imwrite("resize_yolop.jpg", img)
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        count += 1
        cv2.imshow("after mean", img)
        cv2.waitKey(0)
        cv2.imwrite("reduce_mean_yolop.jpg", img)
        return img
        save_path = os.path.join("/root/bdd100k_images/val_yolop_v2", img_name)
        cv2.imwrite(save_path, img)


def preprocess(img_path_list, single_version=False):
        # with      label [1,2,4,0,3,5]
        # without   label [0,1,4,3,2,5]   
        img_front = cv2.imread(img_path_list[0])
        img_front = cv2.resize(img_front, (1024, 576))
        

                  
        img_front_left = cv2.imread(img_path_list[1])
        img_front_left = cv2.resize(img_front_left, (512, 288))
        
        img_front_right = cv2.imread(img_path_list[4])
        img_front_right = cv2.resize(img_front_right, (512, 288))
        
        img_rear = cv2.imread(img_path_list[3])
        img_rear = cv2.resize(img_rear, (512, 288))
        
        img_rear_left = cv2.imread(img_path_list[2])
        img_rear_left = cv2.resize(img_rear_left, (512, 288))
        
        img_rear_right = cv2.imread(img_path_list[5])
        img_rear_right = cv2.resize(img_rear_right, (512, 288))
        
        
        image_norm_front = mmcv.imnormalize(img_front,
                                        mean=np.array([123.675, 116.28, 103.53]),
                                        std=np.array([58.395, 57.12, 57.375]))
            
        
        image_norm_front_left = mmcv.imnormalize(img_front_left,
                                    mean=np.array([123.675, 116.28, 103.53]),
                                    std=np.array([58.395, 57.12, 57.375]))
        
        image_norm_front_right = mmcv.imnormalize(img_front_right,
                                    mean=np.array([123.675, 116.28, 103.53]),
                                    std=np.array([58.395, 57.12, 57.375]))
        
        image_norm_rear = mmcv.imnormalize(img_rear,
                                    mean=np.array([123.675, 116.28, 103.53]),
                                    std=np.array([58.395, 57.12, 57.375]))
        
        image_norm_rear_left = mmcv.imnormalize(img_rear_left,
                                    mean=np.array([123.675, 116.28, 103.53]),
                                    std=np.array([58.395, 57.12, 57.375]))
        
        image_norm_rear_right = mmcv.imnormalize(img_rear_right,
                                    mean=np.array([123.675, 116.28, 103.53]),
                                    std=np.array([58.395, 57.12, 57.375]))


        image_front_data = np.expand_dims(image_norm_front.transpose(2, 0, 1), axis=0)  # np array
        image_front_left_data = np.expand_dims(image_norm_front_left.transpose(2, 0, 1), axis=0)  # np array
        image_front_right_data = np.expand_dims(image_norm_front_right.transpose(2, 0, 1), axis=0)  # np array
        image_rear = np.expand_dims(image_norm_rear.transpose(2, 0, 1), axis=0)  # np array
        image_rear_left = np.expand_dims(image_norm_rear_left.transpose(2, 0, 1), axis=0)  # np array
        image_rear_right = np.expand_dims(image_norm_rear_right.transpose(2, 0, 1), axis=0)  # np array    
        return image_front_data, image_front_left_data, image_front_right_data, image_rear, image_rear_left,image_rear_right
        
def calibration_bev_preprocess(root_dir):
    
    img_dir_list = os.listdir(root_dir)
    for each_dir in img_dir_list:
        each_dir_path = os.path.join(root_dir, each_dir)
        img_name_list = sorted(os.listdir(each_dir_path))
        img_path_list = [os.path.join(each_dir_path, each_img_name) for each_img_name in img_name_list]
        print(img_path_list)
        preprocess(img_path_list)
        

if __name__ == '__main__':
    # main()
    # temp_path = "/Users/bruce/CppProjects/CPlusPlusThings/extensions/opencv_learning/yolop_img/befe15d9-e3db8d6b.jpg"
    # img = cv2.imread("/mnt/share_disk/bruce_cui/Yolop_chip/inference/befe15d9-e3db8d6b.jpg")
    # print(img.shape)
    # calibration_yolop("/root/bdd100k_images/val")
    # calibration_yolop_preprocess("/Users/bruce/CppProjects/CPlusPlusThings/extensions/opencv_learning/yolop_img/")
    calibration_bev_preprocess("/Users/bruce/Downloads/5223_bev_trans/input_img_5223_calibration_data")


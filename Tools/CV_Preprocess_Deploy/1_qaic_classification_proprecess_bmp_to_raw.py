import sys
import argparse
import os
from datetime import datetime
from typing import Tuple
import torch
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


def normalize_img(img):  ### c++: https://blog.csdn.net/wuqingshan2010/article/details/107727909
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    return img


def get_val_transform(size):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # size = (224, 224)
    val_transforms = transforms.Compose([
        transforms.Resize(size[0] + 24),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize]
    )
    return val_transforms


def transform_one_image(image_path, config):
    '''return 1x3x640x640 data'''
    img = default_loader(image_path)
    transform = get_val_transform(config.size)
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
    
        

def convert_img_to_raw(config):

    """_summary_
        一般分类的网络只有一级目录
        多输入网络是二级目录
    """
    if config.nest:
        pass
    else:
        if not os.path.exists(config.dst_img_path): os.makedirs(config.dst_img_path)
        img_name_list = os.listdir(config.ori_img_path)
        count = 0
        
        img_paths = []
        for name in tqdm(img_name_list):
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(config.ori_img_path, name)
                data = transform_one_image(img_path, config)
                dst_path = os.path.join(config.dst_img_path, Path(name).stem + ".raw")
                data.numpy().tofile(dst_path)
                img_paths.append(dst_path)
                count += 1
                if count >= 200: break   

           
        with open(config.img_path_txt, "w") as f:
            for each in img_paths:
                f.write(f"{each}\n")
  

def parse_args():
    """_summary_
    config the parameters 
    Returns:
    """
    parser = argparse.ArgumentParser(description='convert img .png .jpg .bmp  to raw formate')
    # 模型选择
    parser.add_argument('--ori_img_path', 
                        default="/Users/bruce/Downloads/Datasets/object_detection_datasets", 
                        help='input image path')
    
    parser.add_argument('--dst_img_path', 
                        default="/Users/bruce/Downloads/Datasets/calibration_object_detection_datasets",
                        help='dst img path to save')

    parser.add_argument('--size', type=list, default=[640, 640], nargs='+', help='img resize')
    
    parser.add_argument('--img_path_txt', type=str, default="./calibration_yolov5m_without_Norm.txt", help='export txt about img path')
    
    parser.add_argument('--nest', type=bool, default=False, help='export txt about img path')
    
   

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    config = parse_args()
    
    convert_img_to_raw(config)
    # yolov5_preprocess(config)
    # img = cv2.imread("/mnt/share_disk/bruce_cui/Yolop_chip/inference/befe15d9-e3db8d6b.jpg")
    # print(img.shape)
    # calibration_yolop("/root/bdd100k_images/val")
    # calibration_yolop_preprocess("/Users/bruce/CppProjects/CPlusPlusThings/extensions/opencv_learning/yolop_img/")
    # print(img.shape)
import os
import mmcv
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import transforms

input_sizes = [(576, 1024), (576, 1024), (576, 1024), (576, 1024), (576, 1024), (320, 512), (576, 1024), (576, 1024)]
crop_ratios = [None, None, None, None, None, None, (0.0, 0.22, 0.5, 0.72), (0.5, 0.22, 1.0, 0.72)]

def img_proprecessor_mmcv(image_paths):
    imgs = []
    ori_shape = []
    mean = [123.675, 116.28,  103.53 ]
    std = [58.395,   57.12,   57.375]
    
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    to_rgb = False
    to_float32 = True
    
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        ori_shape.append(img.size[::-1])
        crop_ratio = crop_ratios[i]
        if crop_ratio is not None:
            crop = (int(crop_ratio[0]*img.size[0]), 
                    int(crop_ratio[1]*img.size[1]), 
                    int(crop_ratio[2]*img.size[0]), 
                    int(crop_ratio[3]*img.size[1]))
            img = img.crop(crop)
        img = img.resize(input_sizes[i][::-1])
        img = np.array(img)
        if to_float32:
            img = img.astype(np.float32)
        
        img = mmcv.imnormalize(img, mean, std, to_rgb)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        img = torch.from_numpy(img)[None, ...]  # [1, 3, h, w]
        imgs.append(img)
    return imgs 


def img_proprecessor_torch_update_pil(image_paths):
    
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    mean_mmcv = [123.675, 116.28,  103.53 ]
    std_mmcv = [58.395,   57.12,   57.375]
    
    imgs = []
    ori_shape = []
    to_float32 = True  
    to_rgb = False  # 确保使用 RGB 格式，如果需要的话
    
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    mean_mmcv = np.array(mean_mmcv, dtype=np.float32)
    std_mmcv = np.array(std_mmcv, dtype=np.float32)
    
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        
        crop_ratio = crop_ratios[i]
        if crop_ratio is not None:
            crop = (int(crop_ratio[0]*img.size[0]), 
                    int(crop_ratio[1]*img.size[1]), 
                    int(crop_ratio[2]*img.size[0]), 
                    int(crop_ratio[3]*img.size[1]))
            img = img.crop(crop)
        img = img.resize(input_sizes[i][::-1])
        img = np.array(img)
        if to_float32:
            img = img.astype(np.float32)
        
        img /= 255.0
        img = (img - mean) / std
        
        img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0)  # HWC到CHW，然后增加批次维度

        # print("==== pil img and the shape of img ====", img.shape)
        imgs.append(img)
    return imgs



def img_proprecessor_torch_update_opencv_demo_1(image_paths):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]    
    imgs = []
    to_float32 = True

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    for i, img_path in enumerate(image_paths):
        # OpenCV 读取图片，默认为BGR格式
        img = cv2.imread(img_path)
        
        # 获取裁剪比例
        crop_ratio = crop_ratios[i]
        if crop_ratio is not None:
            x1 = int(crop_ratio[0] * img.shape[1])
            y1 = int(crop_ratio[1] * img.shape[0])
            x2 = int(crop_ratio[2] * img.shape[1])
            y2 = int(crop_ratio[3] * img.shape[0])
            img = img[y1:y2, x1:x2]

        if to_float32:
            img = img.astype(np.float32)
            
        # 调整图像尺寸
        img = cv2.resize(img, (input_sizes[i][1], input_sizes[i][0]))  # 注意OpenCV中的顺序是宽度在前，高度在后
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # # 标准化处理
        img /= 255.0
        img -= mean
        img /= std

        # 转换为张量
        img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0)  # HWC到CHW，然后增加批次维度

        # 打印图片的形状
        # print("==== the shape of img ====", img.shape)
        
        imgs.append(img)

    return imgs


def img_proprecessor_torch_update_opencv_demo_2(image_paths):

    mean = [123.675, 116.28,  103.53 ]
    std = [58.395,   57.12,   57.375]
    
    imgs = []
    ori_shape = []
    to_float32 = True

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    for i, img_path in enumerate(image_paths):
        # OpenCV 读取图片，默认为BGR格式
        img = cv2.imread(img_path)
        # 存储原始图片的尺寸
        ori_shape.append((img.shape[0], img.shape[1]))
        # 获取裁剪比例
        crop_ratio = crop_ratios[i]
        if crop_ratio is not None:
            x1 = int(crop_ratio[0] * img.shape[1])
            y1 = int(crop_ratio[1] * img.shape[0])
            x2 = int(crop_ratio[2] * img.shape[1])
            y2 = int(crop_ratio[3] * img.shape[0])
            img = img[y1:y2, x1:x2]

        if to_float32:
            img = img.astype(np.float32)
            
        # 调整图像尺寸
        img = cv2.resize(img, (input_sizes[i][1], input_sizes[i][0]))  # 注意OpenCV中的顺序是宽度在前，高度在后
        
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
    
        # 将BGR转换为RGB
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        

        # 转换为张量
        img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0)  # HWC到CHW，然后增加批次维度

        # 打印图片的形状
        # print("==== the shape of img ====", img.shape)
        
        imgs.append(img)

    return imgs



def custom_sort_key(item):
    custom_sort = ["front_long_camera_record", "front_short_camera_record", "front_middle_camera", "lf_wide_camera", "lr_wide_camera", "rf_wide_camera", "rr_wide_camera"]

    for pattern in custom_sort:
        if pattern in item:
            return custom_sort.index(pattern)
    return len(custom_sort)


if __name__ == "__main__":
    
    # os.listdir()
    image_path = "/mnt/share_disk/bruce_cui/infer_vis/8620_mtn/mtn_pipeline/image_data"
    image_list_data = [
        "front_long_camera_record.jpg",
        "front_short_camera_record.jpg",
        "lf_wide_camera_record.jpg",
        "rf_wide_camera_record.jpg",
        "rear_middle_camera_record.jpg",
        "front_fisheye_camera_record.jpg",
        "left_fisheye_camera_record.jpg",        
        "right_fisheye_camera_record.jpg"
    ]
    
    image_data_path = [os.path.join(image_path, each_name) for each_name in image_list_data]
    sorted(image_data_path, key=custom_sort_key)
    print(image_data_path)
    res_mmcv = img_proprecessor_mmcv(image_data_path)
    res_torch_update_pil = img_proprecessor_torch_update_pil(image_data_path)
    res_torch_update_cv1  = img_proprecessor_torch_update_opencv_demo_1(image_data_path)
    res_torch_update_cv2  = img_proprecessor_torch_update_opencv_demo_2(image_data_path)
    with open("/home/bruce_ultra/workspace/Pytorch_learning/Torch_experiment/mtn_image_cv1_output1.txt", "w") as f:
        for i in res_torch_update_cv1[0].flatten():
            f.write(str(i.item())+"\n");
    
    with open("/home/bruce_ultra/workspace/Pytorch_learning/Torch_experiment/mtn_image_cv1_output7.txt", "w") as f:
        for i in res_torch_update_cv1[7].flatten():
            f.write(str(i.item())+"\n")
    
    # res = res_mmcv[0] - res_torch_update_pil[0]
    # print(res)
    
    
    
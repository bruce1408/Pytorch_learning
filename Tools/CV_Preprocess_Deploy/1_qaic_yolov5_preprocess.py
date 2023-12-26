import cv2
import os
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path


def letterbox(img, new_shape=(640, 640), auto=False, scaleFill=False, scaleUp=True):
    """
    python的信封图片缩放
    :param img: 原图
    :param new_shape: 缩放后的图片
    :param color: 填充的颜色
    :param auto: 是否为自动
    :param scaleFill: 填充
    :param scaleUp: 向上填充
    :return:
    """
    shape = img.shape[:2]  # current shape[height,width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleUp:
        r = min(r, 1.0)  # 确保不超过1
    ration = r, r  # width,height 缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ration = new_shape[1] / shape[1], new_shape[0] / shape[0]
    # 均分处理
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 添加边界
    return img, ration, (dw, dh)


def yolov5_preprocess(config):
    
    if not os.path.exists(config.dst_img_path): os.makedirs(config.dst_img_path)
    img_name_list = os.listdir(config.ori_img_path)
    count = 1
    img_path_list = []
    
    for each_img in tqdm(img_name_list):
        img_path = os.path.join(config.ori_img_path, each_img)
        img = cv2.imread(img_path)
        # ori_img = img.
        img = letterbox(img, new_shape=config.size)[0]  # 图片预处理
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img).astype(np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        assert len(img.shape) == 4
        dst_path = os.path.join(config.dst_img_path, Path(each_img).stem+".raw")
        img.tofile(dst_path)
        img_path_list.append(dst_path)
        if count > 199: break
    
    with open(config.img_path_txt, "w") as f:
        for eachline in img_path_list:
            f.write(eachline+"\n")



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
                        default="/Users/bruce/Downloads/Datasets/calibration_object_detection_datasets_withoutNorm",
                        help='dst img path to save')

    parser.add_argument('--size', type=list, default=[640, 640], nargs='+', help='img resize')
    
    parser.add_argument('--img_path_txt', type=str, default="./calibration_yolov5m_without_Norm.txt", help='export txt about img path')
    
    parser.add_argument('--nest', type=bool, default=False, help='export txt about img path')
    
   

    args = parser.parse_args()
    
    return args


if __name__== "__main__":
    config =  parse_args()
    yolov5_preprocess(config)
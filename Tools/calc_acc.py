import os
from PIL import Image
from tqdm import tqdm
from shutil import copy
from PIL import Image
import cv2
import numpy as np
from termcolor import cprint


def print_info(info, _type=None):
    """_summary_
    usage:
    print_info('=> Total {} images to test.'.format(img_txt_evm_res), ['yellow', 'bold'])
    Args:
        info (_type_): _description_
        _type (_type_, optional): _description_. Defaults to None.
    """
    if _type is not None:
        if isinstance(info, str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info, list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)
        
        
        
def compare_res_with_evm_acc(img_txt_evm_res, img_txt_std_res):
    
    total_pred_num = 0
    total_num = 0
    img_std_res = {}
    with open(img_txt_std_res, "r") as f1:
        for eachline in f1:
            eachline = eachline.strip("\n")
            img_name_perfix, gt_label = eachline.split(":")
            img_std_res[img_name_perfix] = int(gt_label)
    
    with open(img_txt_evm_res) as f:  
        for eachline in f:
            eachline = eachline.strip("\n")
            img_name, label = eachline.split(":")
            
            if(img_std_res[img_name.split(".")[0]]  == int(label)):
                total_pred_num += 1
            total_num += 1

    print_info('------------------------------------------------------------------------------\n'
           '|                    ImageNet Dataset Evaluation Results                      |\n'
           '|                                                                             |\n'
           
           f'| the total img nums is {total_num}, the right predict num is {total_pred_num}, acc is: {total_pred_num / total_num:.3}    |\n'

        #    'the total img nums is {}, the right predict num is {}, acc is: {} \n'.format(total_num, total_pred_num, total_pred_num / total_num )
           '-----------------------------------------------------------------------------', ['yellow', 'bold'])

    # print( 'the total img nums is {}, the right predict num is {}, acc is: {:.3} \n'.format(total_num, total_pred_num, total_pred_num / total_num ))

    

def check_img_dir(folder_path):

    # 获取文件夹中的所有文件
    filenames = os.listdir(folder_path)

    # 遍历文件夹中的每一个文件
    for filename in filenames:
        filepath = os.path.join(folder_path, filename)

        # 检查文件是否是图片格式，这里只检查常见的几种格式，可以根据需要增加
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                # 尝试使用 Pillow 打开图片
                with Image.open(filepath):
                    pass
            except Exception as e:
                print(f"Deleting corrupted image: {filepath} due to {str(e)}")
                os.remove(filepath)


def parse_image(image_path):
    image_data  = cv2.imread(image_path)
    print(image_data)
    return image_data


if __name__ == '__main__':
    
    # ========================================================================
    # 根据板卡上面的结果进行验证
    img_txt_std_res = "/Users/bruce/PycharmProjects/Pytorch_learning/Tools/val_imagenet_label.txt"
    img_txt_evm_res = "/Users/bruce/Downloads/15_Ti_model_files/imagenet_bin_ptq_1127.txt"
    # compare_res_with_evm_acc(img_txt_evm_res, img_txt_std_res)
    # ========================================================================
    # print_info('=> Total {} images to test.'.format(img_txt_evm_res), ['yellow', 'bold'])


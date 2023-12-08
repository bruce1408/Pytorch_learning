import os
from PIL import Image
from tqdm import tqdm
from shutil import copy
from PIL import Image
import cv2
import numpy as np
from termcolor import cprint

        
def jpgToBmp(src_img_path, dst_img_path, nest_dir=False):
    src_img_path_list = []
    if nest_dir:
        img_dir_list = os.listdir(src_img_path)
        for each_dir in img_dir_list:
            img_dir_path = os.path.join(src_img_path, each_dir)
            img_name_list = os.listdir(img_dir_path)
            img_file_path = [os.path.join(img_dir_path, img_name) for img_name in img_name_list ]
            src_img_path_list.append(img_file_path)
            
        for each_list in tqdm(src_img_path_list, desc="convert jpg2bmp"):
            for fileName in each_list:
                if (os.path.splitext(fileName)[1] == '.JPEG') or (os.path.splitext(fileName)[1] == '.jpg'):
                    name = os.path.splitext(fileName)[0]
                    newFileName = name + ".bmp"
                    # img = Image.open(temp_path + "/" + fileName)
                    # img.save(dst_img_path+"/"+newFileName, format="BMP")
                    
                    image = cv2.imread(fileName)
                    cv2.imwrite(newFileName, image)
    else:
        pass


def write_path_to_txt(file_img_path_perfix, img_dir):
    with open("/Users/bruce/Downloads/15_Ti_model_files/val_image_bmp.txt", "w") as f:
        for img_name in sorted(os.listdir(img_dir)):
            # f.write(os.path.join(file_img_path_perfix, img_name)+"\n")
            f.write(img_name+"\n")


def generate_imagenet_val_label(img_dir):
    img_dir_list = os.listdir(img_dir)
    img_dir_list = sorted(img_dir_list)
    # print(img_dir_list)
    
    img_name_to_label = {}
    for index, img_dir_name in enumerate(img_dir_list):
        img_name_list = os.listdir(os.path.join(img_dir, img_dir_name))
        for img_name in img_name_list:
            img_name_to_label[img_name.split(".")[0]] = index

    print(img_name_to_label)
    with open("/Users/bruce/PycharmProjects/Pytorch_learning/Tools/val_imagenet_label.txt", "w") as f:
        for key, value in img_name_to_label.items():
            f.write(key + ":" + str(value) + "\n")




def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info, str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info, list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)
        
        
        
def compare_res_with_evm_ti(img_txt_evm_res, img_txt_std_res):
    
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

    

def jpeg_to_bmp(jpeg_path, bmp_path):
    image = cv2.imread(jpeg_path)
    cv2.imwrite(bmp_path, image)


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
    # val_dir_path = "/Users/bruce/Downloads/Datasets/val"
    # file_img_path_perfix = "/home/root/nfs_dir/cdd"
    # img_dir = "/Users/bruce/Downloads/15_Ti_model_files/val_image_bmp"
    
    # ========================================================================
    src_img_path = "/Users/bruce/Downloads/15_Ti_model_files/laneline_draw_verify_datasets"
    dst_img_path = "/Users/bruce/Downloads/15_Ti_model_files/laneline_draw_verify_datasets"
    jpgToBmp(src_img_path, dst_img_path, nest_dir=True)
    # ========================================================================
    
    # check_img_dir(img_dir)
    # Image.open("/Users/bruce/Downloads/15_Ti_model_files/val_image_bmp/ILSVRC2012_val_00036725.bmp")
    # write_path_to_txt(file_img_path_perfix, img_dir)
    # generate_imagenet_val_label(val_dir_path)
    
    # ========================================================================
    # 根据板卡上面的结果进行验证
    img_txt_std_res = "/Users/bruce/PycharmProjects/Pytorch_learning/Tools/val_imagenet_label.txt"
    img_txt_evm_res = "/Users/bruce/Downloads/15_Ti_model_files/imagenet_bin_ptq_1127.txt"
    # compare_res_with_evm_ti(img_txt_evm_res, img_txt_std_res)
    # ========================================================================
    
    
    # 图片解析
    # path_image = "/Users/bruce/Downloads/15_Ti_model_files/ILSVRC2012_val_00044292.bmp"
    # img_data = parse_image(path_image)
    # # print(img_data.shape)
    
    # image_chw = np.transpose(img_data, (2, 0, 1))
    # # print(image_chw.shape)
    # # print(image_chw[0].shape)

    # h, w = image_chw[0].shape
    # print(h, w)
    # print(image_chw[0][769][479])
    # print(img_data[:][769][479])
    # with open("image_data_00044292_R_python.txt", "w") as f:
    #     for i in range(h):
    #         for j in range(w):
    #             f.write(str(image_chw[2][i][j])+"\n")
                
    
    # from PIL import Image
    # import numpy as np

    # # 定义图片的宽度和高度
    # width = 480
    # height = 770

    # # 打开图片文件
    # image = Image.open("/Users/bruce/Downloads/15_Ti_model_files/ILSVRC2012_val_00044292.bmp")

    # # 读取图片数据
    # image_data = np.array(image)

    # # 输出图片数据到控制台
    # for i in range(height):
    #     for j in range(width):
    #         pixel = image_data[i, j]
    #         print(f"Pixel ({j}, {i}): {pixel[0]}, {pixel[1]}, {pixel[2]}")

    # # 输出图片数据到文本文件
    # with open("/Users/bruce/CppProjects/CPlusPlusThings/extensions/Opencv_learning/1_1_demo_read_img/image_data_1.txt", 'w') as file:
    #     for i in range(height):
    #         for j in range(width):
    #             file.write(f"{image_data[i, j][0]}\n")


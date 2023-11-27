import os
import numpy as np
from tqdm import tqdm


def image_to_binary(input_image_path, output_binary_path):
    try:
        # 以二进制模式读取图像文件
        with open(input_image_path, "rb") as image_file:
            # 读取图像内容
            image_binary = image_file.read()

        # 将图像内容写入二进制文件
        with open(output_binary_path, "wb") as binary_file:
            binary_file.write(image_binary)

        # print(f"\r Image converted to binary and saved to {output_binary_path}")

    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")


def img2bin(img_dir_path, bin_path, nest=False):
    if nest:
        pass
    else:
        img_name_list = os.listdir(img_dir_path)
        for img_name in tqdm(img_name_list, desc="convert img2bin"):
            img_path = os.path.join(img_dir_path, img_name)
            prefix_img_name = os.path.splitext(img_path)[0]

            bin_file_path = prefix_img_name + ".bin"
            image_to_binary(img_path, bin_file_path)
            


if __name__ == "__main__":
    bmp_data_path =  "/Users/bruce/Downloads/15_Ti_model_files/calibration_data_bmp_bin_debug"
    img2bin(bmp_data_path, bmp_data_path, False)



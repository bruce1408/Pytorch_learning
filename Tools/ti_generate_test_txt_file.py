import os
import numpy as np

# 定义排序函数
def sort_key(filename):
    # 从文件名中找到第一个非数字字符的索引
    index = next((i for i, c in enumerate(filename) if not c.isdigit()), len(filename))
    # 返回从找到的索引开始到结尾的子字符串
    return filename[index:]


def generate_test_txt(image_dir_path, test_txt_path):
    dir_list_names = os.listdir(image_dir_path)

    with open(test_txt_path, "w") as f:
        for each_dir_name in dir_list_names:
            image_path = os.path.join(image_dir_path, each_dir_name)
            image_name_list = sorted(os.listdir(image_path), key=sort_key)
            print(image_name_list)
            prefix_name = ["/flag/" + each_dir_name + "/"+each_name for each_name in image_name_list if os.path.splitext(each_name)[1]==".jpg"]
            print(prefix_name)
            # prefix_path = os.path.join(image_dir_path, each_dir_name)
            f.write(" ".join(prefix_name))
            f.write("\n")


def rerank_file_txt(test_txt_path, rerank_txt_path):
    with open(test_txt_path, "r") as f:
        with open(rerank_txt_path, "w") as fr:
            for each_line in f:
                each_data_list = each_line.strip().split(" ")
                print(each_data_list)
                temp_name = each_data_list[0]
                each_data_list[0] = each_data_list[1];
                each_data_list[1] = temp_name
                each_data_list_rerank = [ "/bruce/" + each_data for each_data in each_data_list][0:3]
            
                fr.write(" ".join(each_data_list_rerank))
                fr.write("\n")
            
        
    
if __name__ == "__main__":
    image_dir_path = "/Users/bruce/Downloads/15_Ti_model_files/laneline_draw_verify_datasets"
    test_txt_path = "/Users/bruce/PycharmProjects/Pytorch_learning/Tools/lane_imgs_debug.txt"
    rerank_txt_path = "/Users/bruce/PycharmProjects/Pytorch_learning/Tools/lane_imgs_rerank_debug.txt"
    
    generate_test_txt(image_dir_path, test_txt_path)
    rerank_file_txt(test_txt_path, rerank_txt_path)
    










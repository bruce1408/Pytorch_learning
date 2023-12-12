import os
import re
import shutil
def rename_img_name(img_dir_paths):
    # img_dir_paths = "/Users/bruce/Downloads/15_Ti_model_files/rgb_for_perception_lane/no_compressed_3v"

    img_dir_list = os.listdir(img_dir_paths)
    # print(img_dir_list)

    for each_dir in img_dir_list:
        img_dir_path = os.path.join(img_dir_paths, each_dir)
        img_name_list = os.listdir(img_dir_path)
        # print(img_name_list)
        

        for each_img_name in img_name_list:
            match = re.match(r'^\d+_([^\.]+)\.yuv\.png$', each_img_name)

            if match:
                target_name = match.group(1) + '.png'
                print(f'提取的目标图片名字为: {target_name}')
                
                #os.path.join(path_name,item)表示找到每个文件的绝对路径并进行拼接操作
                os.rename(os.path.join(img_dir_path, each_img_name), os.path.join(img_dir_path, target_name))

            else:
                print('文件名不匹配预期的格式')


def skip_img_count(img_dir_paths):
    img_dir_list = os.listdir(img_dir_paths)
    for each_dir in img_dir_list:
        dir_path = os.path.join(img_dir_paths, each_dir)
        img_list = [each_img for each_img in os.listdir(dir_path)]
        if "front_wide_camera.png" not in img_list:
            shutil.rmtree(dir_path)
         
    
    
if __name__ == "__main__":
    img_dir_paths = "/Users/bruce/Downloads/15_Ti_model_files/rgb_for_perception_lane/compressed_3v"
    skip_img_count(img_dir_paths)

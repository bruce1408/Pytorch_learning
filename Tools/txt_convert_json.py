import json
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
img_info_dict = defaultdict(list)

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y


# 先读取txt文件
with open("/root/Pytorch_learning/Tools/res.txt", "r") as f:
    for eachline in f:
        eachline = eachline.strip("\n").split(",")
        img_info_dict[eachline[0]].append(eachline[1:])
    
    
# 然后生成json文件
jdict = []        
for key, value in tqdm(img_info_dict.items()):
    for bbox_info in value:
        box = list(map(float, bbox_info[2:]))
        # box = convert((720, 1280), box)
        box = xyxy2xywh(box).tolist()
        # print(box)
        jdict.append(
            {
                'images': key,
                'image_id': key.split(".")[0],
                'category_id': 0,
                'bbox': box,
                'score': bbox_info[1]})
        

# 保存到json格式
with open("yolop_pred_coord.json", "w") as f:
    json.dump(jdict, f)
import numpy as np
import pandas as pd

# data_txt = np.loadtxt('/Users/bruce/Downloads/server_downloads/det_result.txt', delimiter=',')
# print(data_txt)
txt_path = '/Users/bruce/Downloads/server_downloads/det_result.txt'
x_center = []
y_center = []
w_center = []
h_center = []
conf_center = []
iou_center = []
with open(txt_path, "r") as f:
    for eachline in f:
        line = eachline.strip("\n").split(",")
        x_center.append(float(line[0]))
        y_center.append(float(line[1]))
        w_center.append(float(line[2]))
        h_center.append(float(line[3]))
        conf_center.append(float(line[4]))
        iou_center.append(float(line[5]))

data_value = {}
data_value["x_center"] = x_center
data_value["y_center"] = y_center
data_value["w_center"] = w_center
data_value["h_center"] = h_center
data_value["conf_center"] = conf_center
data_value["iou_center"] = iou_center
df = pd.DataFrame(data_value)
df.to_csv("./det_result.csv", index=False)
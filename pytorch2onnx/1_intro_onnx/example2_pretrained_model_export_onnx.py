import os
import torch 
from torch import nn 
from torch.nn.functional import interpolate 
import torch.onnx 
import cv2 
import numpy as np 
import requests


def download_model():
    """
    模型和图片进行下载
    """
    urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
            'https://raw.githubusercontent.com/open-mmlab/mmediting/master/tests/data/face/000001.png']
    names = ['./models/srcnn.pth', 'face.png']
    for url, name in zip(urls, names):
        if not os.path.exists(name):
            open(name, 'wb').write(requests.get(url).content)
 
 
class SuperResolutionNet(nn.Module): 
 
    def __init__(self): 
        super().__init__() 
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4) 
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0) 
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2) 
 
        self.relu = nn.ReLU() 
 
    def forward(self, x, upscale_factor): 
        x = interpolate(x, 
                        scale_factor=upscale_factor.item(), 
                        mode='bicubic', 
                        align_corners=False) 
        out = self.relu(self.conv1(x)) 
        out = self.relu(self.conv2(out)) 
        out = self.conv3(out) 
        return out 
 
 
def init_torch_model(): 
    torch_model = SuperResolutionNet() 
 
    state_dict = torch.load("./models/srcnn.pth")['state_dict']
    print(torch.load("./models/srcnn.pth").keys()) 
 
    # Adapt the checkpoint 
    for old_key in list(state_dict.keys()): 
        new_key = '.'.join(old_key.split('.')[1:]) 
        state_dict[new_key] = state_dict.pop(old_key) 
 
    torch_model.load_state_dict(state_dict) 
    torch_model.eval() 
    return torch_model 
 

def prepare_model():
    model = init_torch_model()
    input_img = cv2.imread('models/face.png').astype(np.float32) 
    
    # HWC to NCHW 
    input_img = np.transpose(input_img, [2, 0, 1]) 
    input_img = np.expand_dims(input_img, 0) 
    
    # Inference 
    torch_output = model(torch.from_numpy(input_img), torch.tensor(3)).detach().numpy() 
    
    # NCHW to HWC 
    torch_output = np.squeeze(torch_output, 0) 
    torch_output = np.clip(torch_output, 0, 255) 
    torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8) 
    
    # Show image 
    cv2.imwrite("models/face_torch_2.png", torch_output) 
    return model, input_img, torch_output


# 第一个参数是要转换的模型、接下来是模型的任意一组输入、到处onnx文件的文件名
def export_onnx_model():
    """
        torch.onnx.export PyTorch 自带的把模型转换成 ONNX 格式的函数;
        前三个必选参数：分别是要转换的模型、模型的任意一组输入、导出的 ONNX 文件的文件名
    """
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        torch.onnx.export(model, (x, torch.tensor(3)),
            "./models/srcnn_2.onnx",
            opset_version=11, # 表示onnx算子集的版本
            input_names=['input'],  # 输入
            output_names=['output'])  # 模型输出


if __name__ == "__main__":
    
    # 虽然是改了模型的输入，但是onnx还是只有一个输入
    model, input_img, torch_output = prepare_model()
    export_onnx_model()


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms

 
def img_read_convert_bin(input_path, output_path):
    img = Image.open(input_path)
    img = img.resize((480, 640))
    img = np.array(img)
    
    to_tensor = transforms.Compose([
                    transforms.Resize((480, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    pil_image = Image.fromarray(img)
    img = to_tensor(pil_image)
    img = torch.unsqueeze(img, 0) #给最高位添加一个维度，也就是batchsize的大小

    print(img.shape)
 
    np_img = img.detach().numpy().astype(np.uint8)
    np_img.tofile(output_path)
    


if __name__ == "__main__":
    path_in = "/Users/bruce/Downloads/val/n01440764/ILSVRC2012_val_00039905.JPEG"
    path_out = "./land_small_480_640.bin"
    img_read_convert_bin(path_in, path_out)
    
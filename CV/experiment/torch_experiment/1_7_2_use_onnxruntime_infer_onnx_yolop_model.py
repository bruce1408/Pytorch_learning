import os
import onnx
import torch
import argparse
from tqdm import tqdm
import onnxruntime
import numpy as np
import PIL
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# 1. load config variable and load model from pytorch
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch convert onnx to verficaiton Acc')
    
    # 模型选择
    parser.add_argument('--model_name', default="resnet18", help='model name resnet-50、mobilenet_v2、efficientnet')
    
    # 预处理模型下载地址
    parser.add_argument('--model_file', default="/root/resnet18-f37072fd.pth", help='laod checkpoints from saved models')
    
    # 输入大小
    parser.add_argument('--input_shape', type=list, nargs='+', default=[1, 3, 224, 224])

    # label 地址
    parser.add_argument("--label_path", default="/root/bdd100k_images/val", help="label path")
    
    # onnx 输出地址
    parser.add_argument('--export_path', default="./torch_official_export_model.onnx", help="pth model convert to onnx name")
    
    # 验证集 地址
    parser.add_argument('--data_val', default="/root/val", help="val data")
    
    args = parser.parse_args()
    
    return args


# img preprocessing method
def pre_process_img(image_path):
    
    transform = transforms.Compose([
        transforms.Resize(640),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])
    
    img = PIL.Image.open(image_path).convert("RGB")
    img = transform(img)
    return img


# get the val data and preprocessing 
def load_label_imgs():
    print("laoding the data and preprocessing the image data ...")
    img_path = []
    img_labels = []
    for dir_ in os.listdir(args.data_val):
        img_dir = os.path.join(args.data_val, dir_)
        for img_name in os.listdir(img_dir):
            img_labels.append(dir_)
            img_path.append(os.path.join(img_dir, img_name))

    img_dict = dict()
    label_index = 0
    with open(args.label_path) as f:
        for eachline in f:
            img_dict[label_index] = eachline.split(' ')[0]   
            label_index+=1         
    
    image_datas = [pre_process_img(single_img_path) for single_img_path in tqdm(img_path)]

    return image_datas, img_dict, img_labels


# get the torch official models 
def get_model():
    model = models.__dict__[args.model_name](pretrained=False)
    state_dict = torch.load(args.model_file)
    model.load_state_dict(state_dict)
    return model


# export .pth model to .onnx
def export_onnx(model, input_shape, export_onnx_path):
    model = get_model()
    model.eval()
    torch.onnx.export(model, torch.randn(input_shape), export_onnx_path, input_names=["input"], output_names=["output"], opset_version=11)
    print("onnx model has been transformed!")


# load onnx and val the data with ort
def load_onnx_and_eval(test_images, img_labels):
    
    sess = onnxruntime.InferenceSession(args.export_path)
    correct_count = 0
    total_count = len(test_images)
    print("begin to eval the model...")
    for i in tqdm(range(total_count)):
        input_data = np.expand_dims(test_images[i], axis=0).astype(np.float32)
        output = sess.run(None, {'input': input_data})[0]
        predicted_class = np.argmax(output)
        predict = img_dict[predicted_class]
    
        if predict == img_labels[i]:
            correct_count += 1
            
    accuracy = correct_count / total_count
    print('Classification accuracy:', accuracy)


if __name__ == "__main__":
    args = parse_args()
    
    # 1. 加载数据
    image_datas, img_dict, img_labels = load_label_imgs()
    
    # 2. 加载模型
    model = get_model()
    
    # 3. 导出onnx模型
    export_onnx(model, args.input_shape, args.export_path)
    
    # 4. 用ort进行推理验证
    load_onnx_and_eval(image_datas, img_labels)
    
   
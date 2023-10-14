import onnxruntime
import numpy as np
import os
import numpy as np
import PIL

import torchvision.transforms as transforms


# 加载 ONNX 模型
model_path = "/Users/bruce/Downloads/5223_bev_trans/20230925/modified_modified_modified_0925_mtn_without_linear_dummy_v7.onnx"
model_path = "/Users/bruce/Downloads/5223_bev_trans/20230925/modified_0925_mtn_without_linear_dummy_v8_onnxsim.onnx"
session = onnxruntime.InferenceSession(model_path)

input_data_bev = {}
input_names = []
input_name = session.get_inputs()
input_names = [ each_name.name for each_name in input_name]
input_names = ['left_front_cam','front_cam', 'right_front_cam', 'rear_cam', 'left_rear_cam', 'right_rear_cam']

def pre_process_img(image_path):
    
    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])
    
    img = PIL.Image.open(image_path).convert("RGB")
    img = transform(img)
    return img



# 准备测试数据和标签
test_data = "/Users/bruce/Downloads/5223_bev_trans/5223_virtual_img_with_labels/1676010660085919"  # 将测试数据加载到这里
file_names = sorted(os.listdir(test_data))
correct_predictions = 0
total_samples = len(file_names)

for i in range(total_samples):
    img_path = os.path.join(test_data, file_names[i])
    img_data = pre_process_img(img_path)
    print(img_data.shape)
    # input_name = session.get_inputs()[0].name
    # input_data = np.array(img_data, dtype=np.float32)  # 注意数据类型和形状需要匹配
    input_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    print(input_data.shape)
    # if input_data.shape[1]==576:
    #     input_data["front_cam"] = input_data
    # else:
        
    input_data_bev[input_names[i]] = input_data

# print(input_data_bev)
    
# 进行推理
outputs = session.run(None, input_data_bev)

for out in outputs:
    print(out.shape)
    
#     # 处理模型输出，例如取最大值作为预测标签
#     predicted_label = np.argmax(outputs[0])
#     true_label = test_labels[i]

#     if predicted_label == true_label:
#         correct_predictions += 1

# accuracy = correct_predictions / total_samples
# print(f"Accuracy: {accuracy:.2f}")

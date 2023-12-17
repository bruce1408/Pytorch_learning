import onnxruntime
import numpy as np

# 加载 ONNX 模型
model_path = "/Users/bruce/Downloads/5223_bev_trans/20230811/regnet_cla_224.onnx"
session = onnxruntime.InferenceSession(model_path)

# 准备测试数据和标签
test_data = "/Users/bruce/Downloads/Datasets/val"  # 将测试数据加载到这里

correct_predictions = 0
total_samples = len(test_data)

for i in range(total_samples):
    input_name = session.get_inputs()[0].name
    input_data = np.array(test_data[i], dtype=np.float32)  # 注意数据类型和形状需要匹配

    # 进行推理
    outputs = session.run(None, {input_name: input_data})

    # 处理模型输出，例如取最大值作为预测标签
    predicted_label = np.argmax(outputs[0])
    true_label = test_labels[i]

    if predicted_label == true_label:
        correct_predictions += 1

accuracy = correct_predictions / total_samples
print(f"Accuracy: {accuracy:.2f}")

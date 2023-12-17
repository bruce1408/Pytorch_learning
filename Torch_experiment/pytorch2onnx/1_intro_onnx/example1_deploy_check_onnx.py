import onnx

# 加载onnx模型
onnx_model = onnx.load("./models/srcnn.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")
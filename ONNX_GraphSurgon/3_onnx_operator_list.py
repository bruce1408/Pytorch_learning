import onnx

# 加载ONNX模型
model_path = '/mnt/share_disk/bruce_cui/hm_hp370_bev_v2.6_op16_vt_lane.onnx'  # 请替换成你的ONNX模型文件路径
model = onnx.load(model_path)

# 获取模型中的所有节点
graph = model.graph
nodes = graph.node

# 获取模型中的所有算子,然后保存在set集合里面
print("Operators in the ONNX model:")
operators = set()
for node in nodes:
    operators.add(node.op_type)

print(operators)

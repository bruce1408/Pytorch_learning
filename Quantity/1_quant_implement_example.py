import numpy as np

def quantize(weights, num_bits=8):
    # 将32位浮点数量化为num_bits位整数表示
    qmin = -2**(num_bits - 1)
    qmax = 2**(num_bits - 1) - 1
    scale = max(abs(weights.min()), abs(weights.max())) / qmax
    weights_q = np.round(weights / scale)
    weights_q = np.clip(weights_q, qmin, qmax).astype(np.int8)
    return weights_q, scale

def dequantize(weights_q, scale):
    # 将量化后的整数恢复为32位浮点数
    return weights_q.astype(np.float32) * scale

# 加载原始模型权重
# weights = np.load('model_weights.npy')
weights = np.random.rand(100) * 100

# 定义量化位数
num_bits = 8

# 量化权重
weights_q, scale = quantize(weights, num_bits=num_bits)

# 加载原始模型激活值
# activations = np.load('model_activations.npy')
activations = np.random.rand(100)
# 量化激活值
activations_q = quantize(activations, num_bits=num_bits)

# 使用量化后的权重和激活值进行推理


# 将量化后的权重恢复为32位浮点数
weights_deq = dequantize(weights_q, scale)

# 计算量化误差
weight_error = np.abs(weights - weights_deq).mean()

# 输出量化误差和压缩比
print('Weight quantization error:', weight_error)
print('Weight compression ratio:', weights.size / weights_q.size)

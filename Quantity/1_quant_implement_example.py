import numpy as np


def quantize_simulation():
    
    def quantize(weights, num_bits=8):
        # 将32位浮点数量化为num_bits位整数表示
        qmin = -2 ** (num_bits - 1)
        qmax = 2 ** (num_bits - 1) - 1
        scale_1 = max(abs(weights.min()), abs(weights.max())) / qmax
        scale_2 = (weights.max() - weights.min() ) / (qmax - qmin)
        zero_point = qmax - weights.max() / scale_2
        
        weights_q1 = np.round(weights / scale_1)
        weights_q2 = np.round(weights / scale_2 + zero_point)
        print(weights[0:5])
        print(weights_q1[0:5])
        print(weights_q2[0:5])
        weights_q1 = np.clip(weights_q1, qmin, qmax).astype(np.int8)
        weights_q2 = np.clip(weights_q2, qmin, qmax).astype(np.int8)
        
        return weights_q1, weights_q2, scale_1, scale_2, zero_point


    def dequantize(weights_q, scale, zero_point):
        # 将量化后的整数恢复为32位浮点数
        if zero_point:
            return (weights_q.astype(np.float32) - zero_point) * scale
        else:
            return weights_q.astype(np.float32) * scale


    # 加载原始模型权重
    weights = np.random.randn(1000) * 100

    # 定义量化位数
    num_bits = 8

    # 量化权重
    weights_q1, weights_q2, scale1, scale2, zero_point = quantize(weights, num_bits=num_bits)

    print("scale1 : {} and scale2: {} offset: {} ".format( scale1, scale2, zero_point))
    # 将量化后的权重恢复为32位浮点数
    weights_deq1 = dequantize(weights_q1, scale1, None)
    weights_deq2 = dequantize(weights_q2, scale2, zero_point)

    # 计算量化误差
    weight_error1 = np.abs(weights - weights_deq1).mean()
    weight_error2 = np.abs(weights - weights_deq2).mean()

    # 输出量化误差和压缩比
    print('Weight1 quantization error:', weight_error1)
    print('Weight2 quantization error:', weight_error2)
    # print('Weight compression ratio:', weights.size / weights_q1.size)


quantize_simulation()
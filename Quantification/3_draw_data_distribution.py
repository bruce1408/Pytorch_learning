import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

# 示例数据
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data = np.random.randn(1000)*100

def draw_data_distribution(data):
    # 计算最小值和最大值
    min_value = np.min(data)
    max_value = np.max(data)

    # 计算数据范围
    data_range = max_value - min_value

    # 打印最小值、最大值和数据范围
    print("最小值:", min_value)
    print("最大值:", max_value)
    print("数据范围:", data_range)

    # 绘制直方图
    plt.hist(data, bins=20)
    plt.xlabel("数据")
    plt.ylabel("频数")
    plt.title("数据直方图")
    plt.show()

    # # 绘制箱线图
    # plt.boxplot(data)
    # plt.ylabel("数据")
    # plt.title("数据箱线图")
    # plt.show()


draw_data_distribution(data)
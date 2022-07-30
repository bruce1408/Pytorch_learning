# coding=utf-8
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from scipy.optimize import leastsq
from scipy import stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import CubicSpline
import scipy as sp
import math

# if __name__ == '__main__':
'''
画图
'''






# 5.绘图
# 5.1 绘制正态分布概率密度函数
# mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
# mpl.rcParams['axes.unicode_minus'] = False
# mu = 0
# sigma = 1
# x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 51)
# y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
# print x.shape
# print 'x = \n', x
# print y.shape
# print 'y = \n', y
# plt.figure(facecolor='w')
# plt.plot(x, y, 'ro-', linewidth=2)
# # plt.plot(x, y, 'r-', x, y, 'go', linewidth=2, markersize=8)
# plt.xlabel('X', fontsize=15)
# plt.ylabel('Y', fontsize=15)
# plt.title(u'高斯分布函数', fontsize=18)
# plt.grid(True)
# plt.show()

# 5.2 损失函数：Logistic损失(-1,1)/SVM Hinge损失/ 0/1损失
# plt.figure(figsize=(10,8))
# x = np.linspace(start=-2, stop=3, num=1001, dtype=np.float)
# y_logit = np.log(1 + np.exp(-x)) / math.log(2)
# y_boost = np.exp(-x)
# y_01 = x < 0
# y_hinge = 1.0 - x
# y_hinge[y_hinge < 0] = 0
# plt.plot(x, y_logit, 'r-', label='Logistic Loss', linewidth=2)
# plt.plot(x, y_01, 'g-', label='0/1 Loss', linewidth=2)
# plt.plot(x, y_hinge, 'b-', label='Hinge Loss', linewidth=2)
# plt.plot(x, y_boost, 'm--', label='Adaboost Loss', linewidth=2)
# plt.grid()
# plt.legend(loc='upper right')
# plt.savefig('1.png')
# plt.show()

# 5.3 x^x
# plt.figure(facecolor='w')
# x = np.linspace(-1.3, 1.3, 101)
# y = f(x)
# plt.plot(x, y, 'g-', label='x^x', linewidth=2)
# plt.grid()
# plt.legend(loc='upper left')
# plt.show()

# 5.4 胸型线
# x = np.arange(1, 0, -0.001)
# y = (-3 * x * np.log(x) + np.exp(-(40 * (x - 1 / np.e)) ** 4) / 25) / 2
# plt.figure(figsize=(5,7), facecolor='w')
# plt.plot(y, x, 'r-', linewidth=2)
# plt.grid(True)
# plt.title(u'胸型线', fontsize=20)
# # plt.savefig('breast.png')
# plt.show()

# 5.5 心形线
# t = np.linspace(0, 2*np.pi, 100)
# x = 16 * np.sin(t) ** 3
# y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
# plt.plot(x, y, 'r-', linewidth=2)
# plt.grid(True)
# plt.show()

# # 5.6 渐开线
# t = np.linspace(0, 50, num=1000)
# x = t*np.sin(t) + np.cos(t)
# y = np.sin(t) - t*np.cos(t)
# plt.plot(x, y, 'r-', linewidth=2)
# plt.grid()
# plt.show()

# Bar
# x = np.arange(0, 10, 0.1)
# y = np.sin(x)
# plt.bar(x, y, width=0.04, linewidth=0.2)
# plt.plot(x, y, 'r--', linewidth=2)
# plt.title(u'Sin曲线')
# plt.xticks(rotation=-60)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid()
# plt.show()

# 6. 概率分布
# 6.1 均匀分布
x = np.random.rand(10000)
t = np.arange(len(x))
# plt.hist(x, 30, color='m', alpha=0.5, label=u'均匀分布')
plt.plot(t, x, 'g.', label='均匀分布')
plt.legend(loc='upper left')
plt.grid()
plt.show()

# # 6.2 验证中心极限定理
# t = 1000
# a = np.zeros(10000)
# for i in range(t):
#     a += np.random.uniform(-5, 5, 10000)
# a /= t
# plt.hist(a, bins=30, color='g', alpha=0.5, normed=True, label=u'均匀分布叠加')
# plt.legend(loc='upper left')
# plt.grid()
# plt.show()

# 6.21 其他分布的中心极限定理
# lamda = 7
# p = stats.poisson(lamda)
# y = p.rvs(size=1000)
# mx = 30
# r = (0, mx)
# bins = r[1] - r[0]
# plt.figure(figsize=(15, 8), facecolor='w')
# plt.subplot(121)
# plt.hist(y, bins=bins, range=r, color='g', alpha=0.8, normed=True)
# t = np.arange(0, mx+1)
# plt.plot(t, p.pmf(t), 'ro-', lw=2)
# plt.grid(True)
#
# N = 1000
# M = 10000
# plt.subplot(122)
# a = np.zeros(M, dtype=np.float)
# p = stats.poisson(lamda)
# for i in np.arange(N):
#     a += p.rvs(size=M)
# a /= N
# plt.hist(a, bins=20, color='g', alpha=0.8, normed=True)
# plt.grid(b=True)
# plt.show()

# 6.3 Poisson分布
# x = np.random.poisson(lam=5, size=10000)
# print x
# pillar = 15
# a = plt.hist(x, bins=pillar, normed=True, range=[0, pillar], color='g', alpha=0.5)
# plt.grid()
# plt.show()
# print a
# print a[0].sum()

# # 6.4 直方图的使用
# mu = 2
# sigma = 3
# data = mu + sigma * np.random.randn(1000)
# h = plt.hist(data, 30, normed=1, color='#FFFFA0')
# x = h[1]
# y = norm.pdf(x, loc=mu, scale=sigma)
# plt.plot(x, y, 'r-', x, y, 'ro', linewidth=2, markersize=4)
# plt.grid()
# plt.show()


# # 6.5 插值
# rv = poisson(5)
# x1 = a[1]
# y1 = rv.pmf(x1)
# itp = BarycentricInterpolator(x1, y1)  # 重心插值
# x2 = np.linspace(x.min(), x.max(), 50)
# y2 = itp(x2)
# cs = sp.interpolate.CubicSpline(x1, y1)       # 三次样条插值
# plt.plot(x2, cs(x2), 'm--', linewidth=5, label='CubicSpine')           # 三次样条插值
# plt.plot(x2, y2, 'g-', linewidth=3, label='BarycentricInterpolator')   # 重心插值
# plt.plot(x1, y1, 'r-', linewidth=1, label='Actural Value')             # 原始值
# plt.legend(loc='upper right')
# plt.grid()
# plt.show()

# 6.6 Poisson分布
# size = 1000
# lamda = 5
# p = np.random.poisson(lam=lamda, size=size)
# plt.figure()
# plt.hist(p, bins=range(3 * lamda), histtype='bar', align='left', color='r', rwidth=0.8, normed=True)
# plt.grid(b=True, ls=':')
# # plt.xticks(range(0, 15, 2))
# plt.title('Numpy.random.poisson', fontsize=13)
#
# plt.figure()
# r = stats.poisson(mu=lamda)
# p = r.rvs(size=size)
# plt.hist(p, bins=range(3 * lamda), color='r', align='left', rwidth=0.8, normed=True)
# plt.grid(b=True, ls=':')
# plt.title('scipy.stats.poisson', fontsize=13)
# plt.show()

#######################################################################################################
#######################################################################################################


# 7. 绘制三维图像
# x, y = np.mgrid[-3:3:7j, -3:3:7j]
# print x
# print y
# u = np.linspace(-3, 3, 101)
# x, y = np.meshgrid(u, u)
# print x
# print y
# z = x*y*np.exp(-(x**2 + y**2)/2) / math.sqrt(2*math.pi)
# # z = x*y*np.exp(-(x**2 + y**2)/2) / math.sqrt(2*math.pi)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.plot_surface(x, y, z, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0.1)  #
# ax.plot_surface(x, y, z, rstride=3, cstride=3, cmap=cm.gist_heat, linewidth=0.5)
# plt.show()
# # cmaps = [('Perceptually Uniform Sequential',
# #           ['viridis', 'inferno', 'plasma', 'magma']),
# #          ('Sequential', ['Blues', 'BuGn', 'BuPu',
# #                          'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
# #                          'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
# #                          'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
# #          ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
# #                              'copper', 'gist_heat', 'gray', 'hot',
# #                              'pink', 'spring', 'summer', 'winter']),
# #          ('Diverging', ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
# #                         'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
# #                         'seismic']),
# #          ('Qualitative', ['Accent', 'Dark2', 'Paired', 'Pastel1',
# #                           'Pastel2', 'Set1', 'Set2', 'Set3']),
# #          ('Miscellaneous', ['gist_earth', 'terrain', 'ocean', 'gist_stern',
# #                             'brg', 'CMRmap', 'cubehelix',
# #                             'gnuplot', 'gnuplot2', 'gist_ncar',
# #                             'nipy_spectral', 'jet', 'rainbow',
# #                             'gist_rainbow', 'hsv', 'flag', 'prism'])]

# 8.1 scipy
# 线性回归例1
# x = np.linspace(-2, 2, 50)
# A, B, C = 2, 3, -1
# y = (A * x ** 2 + B * x + C) + np.random.rand(len(x)) * 0.95
#
# t = leastsq(residual, [0, 0, 0], args=(x, y))
# theta = t[0]
# print '真实值：', A, B, C
# print '预测值：', theta
# y_hat = theta[0] * x ** 2 + theta[1] * x + theta[2]
# plt.plot(x, y, 'r-', linewidth=2, label=u'Actual')
# plt.plot(x, y_hat, 'g-', linewidth=2, label=u'Predict')
# plt.legend(loc='upper left')
# plt.grid()
# plt.show()

# # 线性回归例2
# x = np.linspace(0, 5, 100)
# a = 5
# w = 1.5
# phi = -2
# y = a * np.sin(w*x) + phi + np.random.rand(len(x))*0.5
# t = leastsq(residual2, [3, 5, 1], args=(x, y))
# theta = t[0]
# print '真实值：', a, w, phi
# print '预测值：', theta
# y_hat = theta[0] * np.sin(theta[1] * x) + theta[2]
# plt.plot(x, y, 'r-', linewidth=2, label='Actual')
# plt.plot(x, y_hat, 'g-', linewidth=2, label='Predict')
# plt.legend(loc='lower left')
# plt.grid()
# plt.show()

# # 8.2 使用scipy计算函数极值
# a = opt.fmin(f, 1)
# b = opt.fmin_cg(f, 1)
# c = opt.fmin_bfgs(f, 1)
# print a, 1/a, math.e
# print b
# print c

# #
# ===================================== loss draw 一幅图片中同时画两个曲线  =========================================
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# fr1 = open('C:\\Users\\bruce\\Desktop\\log_inception_v4.txt','r+')
# fr2 = open('C:\\Users\\bruce\\Desktop\\log_resnet.txt','r+')
# def listx_y(filename):
#     xdata = []
#     ydata = []
#     for each_line1 in filename:
#         each_line1 = each_line1.rstrip('\n')
#         xdata.append(int(each_line1.split(' ')[0]))
#         ydata.append(float(each_line1.split(' ')[1]))
#     print 'the %s is '%filename,xdata
#     print 'the %s is '%filename,ydata
#     return xdata, ydata
#
#
# listX1, listY1 = listx_y(fr1)
# listX2, listY2 = listx_y(fr2)
#
#
# fig = plt.figure(figsize=(10,8),facecolor='white')
# ax = fig.add_subplot(111)
# class1 = plt.plot(listX1, listY1,'r-',label='inception loss',linewidth=1.5)
# class2 = plt.plot(listX2, listY2,'g-',label='resnet loss',linewidth=1.5)
# plt.legend(loc='upper right')
# plt.show()
#
# fr1.close()
# fr2.close()

# ============================================画带有标签的散点图=============================================

__author__ = 'Bruce Cui'
import os
import pandas as pd
# import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # 只显示 Error

# import re
# from string import punctuation
# import knn1 # knn1 在D盘的那个目录machine_learning 目录下面，可以找到然后加进来即可
# import mglearn
# import operator as op
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# np.set_printoptions(suppress=True,threshold=np.NaN,linewidth=150)
# mpl.rcParams['font.sans-serif'] = [u'SimHei']  # FangSong/黑体 FangSong/KaiTi
# mpl.rcParams['axes.unicode_minus'] = False
#
#
# # --------------------------------------------第一类散点图(强烈推荐一，代码最少，表现丰富)--------------------------------
# plt.figure(figsize=(10,8),facecolor='white')
# mglearn.discrete_scatter(returnMat[:, 0], returnMat[:, 1], classLabelVector)
# plt.legend([u'第0类', u"Class 1",u'第二类'], loc=4)
# plt.xlabel(u'玩视屏游戏所耗费的百分比',fontsize=17)
# plt.ylabel(u'每周消耗的冰淇淋公升数',fontsize=17)
# plt.show()
# ----------------------------------------------------- the end -------------------------------------------



# --------------------------------------------(第一列和第二列三分类的点)(方法二)------------------------------------
# returnMat,classLabelVector = knn1.file2matrx('D:\\all_example\\machinelearninginaction\\Ch02\\datingTestSet2.txt')
# x_1 = []; y_1 = []
# x_2 = []; y_2 = []
# x_3 = []; y_3 = []
#
# for i in range(0,1000):
#     if classLabelVector[i]==1:
#         x_1.append(returnMat[i][0])
#         y_1.append(returnMat[i][1])
#     elif classLabelVector[i]==2:
#         x_2.append(returnMat[i][0])
#         y_2.append(returnMat[i][1])
#     else:
#         x_3.append(returnMat[i][0])
#         y_3.append(returnMat[i][1])
#
#
# # -------------------------------------------------三分类的散点图----------------------------------------------------
# fig = plt.figure(figsize=(10,8),facecolor='white')
# ax = fig.add_subplot(111)
# class1 = ax.scatter(x_1, y_1,s=60,c='r')
# class2 = ax.scatter(x_2,y_2,s=30,c='b')
# class3 = ax.scatter(x_3,y_3,s=50,c='g')
#
# # ax.scatter(returnMat[:,1],returnMat[:,2])
# plt.xlabel(u'每年获得的飞行的里程数',fontsize=17)
# plt.ylabel(u'玩视屏游戏所耗费的百分比',fontsize=17)
# # ax.scatter(returnMat[:,0],returnMat[:,1],15.0*np.array(classLabelVector),15.0*np.array(classLabelVector))
# plt.legend([class1,class2,class3],[u'不喜欢',u'一般喜欢',u'非常喜欢'],loc='upper right')
# plt.savefig('C:\\Users\\bruce\\Desktop\\scatter.jpg',format='jpg')
# plt.show()
# --------------------------------------------三分类的散点图结束(方法二)------------------------------------------



# -----------------------------------------------画散点图方法三(第二列和第三列)------------------------------------
# returnMat,classLabelVector = knn.file2matrx('D:\\all_example\\machinelearninginaction\\Ch02\\datingTestSet2.txt')
# fig = plt.figure(figsize=(10,8),facecolor='white')
# ax = fig.add_subplot(111)
# # ax.scatter(returnMat[:,1],returnMat[:,2])
# plt.xlabel(u'玩视屏游戏所耗费的百分比',fontsize=17)
# plt.ylabel(u'每周消耗的冰淇淋公升数',fontsize=17)
# ax.scatter(returnMat[:,1],returnMat[:,2],15.0*np.array(classLabelVector),15.0*np.array(classLabelVector))
# # ax.scatter(returnMat[:,1],returnMat[:,2])
# plt.savefig('C:\\Users\\bruce\\Desktop\\scatter.jpg',format='jpg')
# # plt.legend()
# plt.show()
# ---------------------------------------------------方法三结束----------------------------------------------------



# --------------------------------------------------画散点图方法四----------------------------------------------
# returnMat,classLabelVector = knn1.file2matrx('D:\\all_example\\machinelearninginaction\\Ch02\\datingTestSet2.txt')
# # fig = plt.figure(figsize=(10,8),facecolor='white')
# plt.figure(figsize=(10,8),facecolor='white')
# # fig.add_subplot(111)
# plt.scatter(returnMat[:,0],returnMat[:,1])
# plt.xlabel(u'玩视屏游戏所耗费的百分比',fontsize=17)
# plt.ylabel(u'每周消耗的冰淇淋公升数',fontsize=17)
# # plt.scatter(returnMat[:,0],returnMat[:,1],15.0*np.array(classLabelVector),15.0*np.array(classLabelVector))
# plt.savefig('C:\\Users\\bruce\\Desktop\\scatter.jpg',format='jpg')
# plt.show()
# =====================================================方法四结束================================================




# ============================================== 利用python可视化数据 =============================================

'''
这里的所有画图都是基于matplotlib的模块来画图的
我们有两种分开表格绘图的方式，一种就是    fig = plt.figure() # 创建一个新的figure图片
还有一种是                               fig,axes = plt.subplots(2, 2,sharex=True,sharey=True)



'''
# ------------------------------------- 创建一个新的figure图片 ----------------------------------
# fig = plt.figure()
# ax1 = fig.add_subplot(2,2,1)
# ax2 = fig.add_subplot(2,2,2)
# ax3 = fig.add_subplot(2,2,3)
#
# plt.plot(np.random.randn(50).cumsum(),'k--')
# ax1.hist(np.random.randn(100),bins=20,color='k',alpha=0.3)          # 直方图
# ax2.scatter(np.arange(30),np.arange(30)+3*np.random.randn(30))      # 散点图
# ax3.plot(np.random.randn(50).cumsum(),'k--')                        # 和 plt.plot功能是一样的
# plt.show()

# -------------------------------------- 第二种分图的方式 ---------------------------------------
# fig,axes = plt.subplots(2, 2,sharex=True,sharey=True)
# for i in range(2):
#     for j in range(2):
#         axes[i,j].hist(np.random.randn(500),bins=50,color='k',alpha=0.3)
# plt.subplots_adjust(wspace=0,hspace=0)
# plt.show()


# ----------------------------------- 一个图中不同的形状的图形 -----------------------------------
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(np.random.randn(1000).cumsum(),'k',label='one')
# ax.plot(np.random.randn(1000).cumsum(),'r--',label='two')
# ax.plot(np.random.randn(1000).cumsum(),'g.-',label='three')
# ax.set_xlabel('stage')
# ax.set_ylabel('log')
# ax.set_title('my first pic')
# ax.legend(loc='best')
# plt.show()

'''

以上都是基于 matplotlib 的模块的方法。下面是利用pandas本身的属性来解决问题

'''





# s = pd.DataFrame(np.random.randn(10,4).cumsum(0),columns=['a','b','c','d'],index=np.arange(0,100,10))
# s.plot()
# plt.show()


'''
# Series和DataFram里面的都有axes[0]和axes[1]，分开两个子图，否则，那么就是单独的两个图像。

'''
# fig,axes = plt.subplots(2,1)
# data = pd.Series(np.random.randn(16),index=list('abcdefghigklmnop'))
# data.plot(kind='bar',ax=axes[0],color='b',alpha=0.3)
# data.plot(kind='barh',ax=axes[1],color='r',alpha=0.5)
# # plt.savefig('C:\\Users\\bruce\\Desktop\\1.png',dpi=400,bbox_inches='tight') #分辨率是400，可以减除空白部分
# plt.show()

# fig,axes = plt.subplots(3,1)
# df = pd.DataFrame(np.random.randn(6,4),index=['a','b','c','d','e','f'],
#                   columns=pd.Index(['one','two','three','four'],name='Genus'))
# df.plot(kind='bar',ax = axes[0])
# df.plot(kind='barh',ax = axes[1],stacked=True,alpha=0.5)  # 堆叠
# df.plot(kind='barh',ax = axes[2],alpha=0.5)
# # df.plot(kind='barh',stacked=True,alpha=0.5)# 没有axes的话就是单独的一个图
# plt.show()

## 直方图
# x = np.random.normal(0,1,size=200)
# y = np.random.normal(10,2,size=200)
# value = pd.Series(np.concatenate([x,y]))
# value.hist(bins=20,alpha=0.4,color='k',normed=True)
# value.plot(color='b',kind='kde')
# plt.show()


datadraw = pd.read_csv('D:\\data\\Datasets\\tc\\ccf_first_round_user_shop_behavior.csv')
print(datadraw.info())
print(datadraw[['time_stamp','longitude','latitude']][:10])




# np.random.seed(0)
# N= 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# c = np.random.rand(N)
#
# area = np.pi*(15*np.random.rand(N))**2
# plt.scatter(x,y,s = area,c = c, alpha=0.5)
# plt.colorbar()
# plt.show()

###---------------------------------- 多种颜色的处理方法 --------------------------------------------
'''
https://www.cnblogs.com/darkknightzh/p/6117528.html
上面的博客地址详细的颜色配置说明方法；
选择一个合适的起点start，stop 范围，然后选择颜色变化范围，最后调用列表colors来实现每种颜色

'''
from matplotlib import cm
start = 0.6
stop = 1.0
number_of_lines = 2200
cm_subsection = np.linspace(start, stop, number_of_lines)
colors = [cm.jet(x) for x in cm_subsection ]

# marker	description
# ”.”	point
# ”,”	pixel
# “o”	circle
# “v”	triangle_down
# “^”	triangle_up
# “<”	triangle_left
# “>”	triangle_right
# “1”	tri_down
# “2”	tri_up
# “3”	tri_left
# “4”	tri_right
# “8”	octagon
# “s”	square
# “p”	pentagon
# “*”	star
# “h”	hexagon1
# “H”	hexagon2
# “+”	plus
# “x”	x
# “D”	diamond
# “d”	thin_diamond
# “|”	vline
# “_”	hline
# TICKLEFT	tickleft
# TICKRIGHT	tickright
# TICKUP	tickup
# TICKDOWN	tickdown
# CARETLEFT	caretleft
# CARETRIGHT	caretright
# CARETUP	caretup
# CARETDOWN	caretdown



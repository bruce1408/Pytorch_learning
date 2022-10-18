import sys
import cv2
imgpath = "/home/bruce/Downloads/nms.jpg"
# 读取照片，这里要根据具体情况设置路径
im = cv2.imread(imgpath)
print(im.shape)
# 重置图片大小，高设置为 400，保持高、宽比例
newHeight = 400
newWidth = int(im.size()[1] * 400 / im.size()[0])
im = cv2.resize(im, (newWidth, newHeight))

# 创建 Selective Search Segmentation 对象
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# 添加待处理的图片
ss.setBaseImage(im)

# 可以选择快速但是低 recall 的方式
# 这里的 recall 指的是选择出来的 region 是否包含了所有应该包含的区域。recall 越高越好
# ss.switchToSelectiveSearchFast()

# 也可以选择慢速但是高 recall 的方式
ss.switchToSelectiveSearchQuality()


# 进行 region 划分，输出得到的 region 数目
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))

# 设定要显示的 region 数目
numShowRects = 100

# 可以通过按键逐步增加或者减少显示的 region 数目
increment = 50

while True:
    # 不要在原图上画 bounding box，而是复制一个新图
    imOut = im.copy()

    # 遍历 regions
    for i, rect in enumerate(rects):
        # 通过 bounding box 显示出指定数量的 region
        if (i < numShowRects):
            x, y, w, h = rect  # bounding box 左上角坐标 x,y, 以及 box 的宽和高
            cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1) # 绿色 box，线宽为 1
        else:
            break

    # 显示图片+bbox
    cv2.imshow("Output", imOut)

    # 接收按键输入
    k = cv2.waitKey(0) & 0xFF

    # “m” 键 is pressed
    if k == 109:
        # 增加显示的 bbox 数目
        numShowRects += increment
    # “l” 键 is pressed
    elif k == 108 and numShowRects > increment:
        # 减少显示的 bbox 数目
        numShowRects -= increment
    # “q” 键 is pressed
    elif k == 113:
        break
# close image show window
cv2.destroyAllWindows()
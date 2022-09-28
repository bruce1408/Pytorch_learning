import torch

pthfile = '/home/cuidongdong/BEVDet/outputs/bevdet-r50/epoch_24.pth'  # faster_rcnn_ckpt.pth
net = torch.load(pthfile, map_location=torch.device('cpu'))  # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上

# print(type(net))  # 类型是 dict
# print(len(net))   # 长度为 3，即存在3个 key-value 键值对
print(net['state_dict'].__len__())
print(net['state_dict'].keys().__len__())
print(net['state_dict'].keys())

# count = 0
# for (layername, param) in zip(net["state_dict"].keys(), net["state_dict"]):
#     if layername.count("img_backbone"):
#         print(layername)
#         print(net["state_dict"][layername].numel())
#         count += net["state_dict"][layername].numel()
#         # print(net[layername])
#         # print(net.parameters()[layername])
#
# print(count)


def calculateParameters(modelpath, layerprefix=None, wholeModel=True):

    if layerprefix !=None:
        wholeModel = False

    net = torch.load(modelpath, map_location=torch.device('cpu'))  # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上

    # 先打印出模型的各个层的名称，方便后面进行某个层的提取
    print(net['state_dict'].keys())

    # 模型的参数名和模型的参数应该是一一对应的
    assert (net['state_dict'].__len__()) == net['state_dict'].keys().__len__()
    total = 0
    for (layername, param) in zip(net["state_dict"].keys(), net["state_dict"]):
        if wholeModel==False:
            if layername.count(layerprefix):
                # print(layername)
                # print(net["state_dict"][layername].numel())
                total += net["state_dict"][layername].numel()
        else:
            total += net["state_dict"][layername].numel()
    return total


param = calculateParameters(pthfile, "img_backbone")
print(param)

param1 = calculateParameters(pthfile, "img_bev_encoder_backbone")
print(param1)

param2 = calculateParameters(pthfile)
print(param2)



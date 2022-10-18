import torch

pthfile = '/home/cuidongdong/epoch_8.pth'  # faster_rcnn_ckpt.pth
net = torch.load(pthfile, map_location=torch.device('cpu'))  # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上

# print(type(net))  # 类型是 dict
# print(len(net))   # 长度为 3，即存在3个 key-value 键值对
print(net['state_dict'].__len__())
print(net['state_dict'].keys().__len__())
print(net['state_dict'].keys())

print(net.keys())
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


def calculateParameters1(modelpath, layerprefix=None, wholeModel=True):

    if layerprefix !=None:
        wholeModel = False

    net = torch.load(modelpath, map_location=torch.device('cpu'))  # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上
    print('='*12, net['optimizer']['param_groups'][0])
    print('='*12+"param_group", net['optimizer']['param_groups'])
    net["optimizer"]["param_groups"][0]["lr"] = 2e-4
    print('='*12+"after", net['optimizer']['param_groups'][0])

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



def calculateParameters(modelpath, layerprefix=None, wholeModel=True):

    if layerprefix !=None:
        wholeModel = False

    net = torch.load(modelpath, map_location=torch.device('cpu'))  # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上
    print('='*12, net['optimizer']['param_groups'][0])
    # print('='*12+"param_group", net['optimizer']['param_groups'])
    # net["optimizer"]["param_groups"][0]["lr"] = 2e-4
    # print('='*12+"after", net['optimizer']['param_groups'][0])

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
#
# param1 = calculateParameters(pthfile, "img_bev_encoder_backbone")
# print(param1)

# param2 = calculateParameters1(pthfile)
# print(param2)
# net["optimizer"]["param_groups"][0]["lr"] = 2e-5

# torch.save({
#     "state_dict": net['state_dict'],
#     "meta": net['meta'],
#     "optimizer": net['optimizer']
# }, "/datasets/cdd_data/lr_change_mobilenetv2.pth")


# net = torch.load("/datasets/cdd_data/lr_change_mobilenetv2.pth", map_location=torch.device('cpu'))  # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上
# print('='*12, net['optimizer']['param_groups'][0])
# print('='*12+"param_group", net['optimizer']['param_groups'][0]['lr'])
# print(net)

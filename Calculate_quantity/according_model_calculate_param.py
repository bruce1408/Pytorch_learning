import torch

pthfile = '/home/cuidongdong/pytorch_learning/CV/Classify/resnet50/outputs/resnet50.pth'  # faster_rcnn_ckpt.pth
net = torch.load(pthfile, map_location=torch.device('cpu'))  # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上


# 772
# net['state_dict'].__len__()) == net['state_dict'].keys().__len__()

# 只有3个key，分别是meta, state_dict, optimizer
print(net.keys())

print(net['state_dict'].keys())
print(net['optimizer'].keys())
print(net['scheduler']['_last_lr'])


def calculateParameters(modelpath, layerprefix=None, wholeModel=True):

    if layerprefix !=None:
        wholeModel = False

    net = torch.load(modelpath, map_location=torch.device('cpu'))  # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上

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


# param = calculateParameters(pthfile, "img_backbone")
# param = calculateParameters(pthfile, "img_bev_encoder_neck")
# print(param)
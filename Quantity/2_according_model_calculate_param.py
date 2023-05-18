import torch
from torchsummary import summary
from thop import profile
import torchvision
from torchvision.models import vgg11_bn

# pthfile = '/home/cuidongdong/pytorch_learning/CV/Classify/resnet50/outputs/resnet50.pth'  # faster_rcnn_ckpt.pth
# net = torch.load(pthfile, map_location=torch.device('cpu'))  # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上


def calculateBEVParameters(modelpath, layerprefix=None, wholeModel=True):
    """
    # net['state_dict'].__len__()) == net['state_dict'].keys().__len__()
    # 只有3个key: 分别是meta, state_dict, optimizer
    :param modelpath:
    :param layerprefix:
    :param wholeModel:
    :return:
    """

    if layerprefix !=None:
        wholeModel = False

    net = torch.load(modelpath, map_location=torch.device('cpu'))  # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上

    print("has keys: ", net.keys())
    for key in net.keys():
        print(key, net[key].keys())
        print(net['optimizer'].keys())
        # print(net['scheduler']['_last_lr'])
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


if __name__ == "__main__":
    print('==> Building model..')
    model = torchvision.models.alexnet(pretrained=False)

    dummy_input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    # param = calculateBEVParameters(pthfile, "img_backbone")
    # param = calculateBEVParameters(pthfile, "img_bev_encoder_neck")
    # print(param)
    path = "/home/cuidongdong/backbone_evaluation/bevdepth4d-r50.pth"
    param = calculateBEVParameters(path)
    print(param/1e6)

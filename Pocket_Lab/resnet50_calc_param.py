import torch.cuda
from torchvision.models import resnet50
from torchsummary import summary
net = resnet50(pretrained="/datasets/cdd_data/resnet50-0676ba61.pth")
if torch.cuda.is_available():
    summary(net.cuda(), (3, 224, 224))
else:
    summary(net, (3, 224, 224))


def resnet_calc_param():
    from torchvision.models import resnet50
    from thop import profile
    import torch
    model = resnet50(pretrained=False)
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input, ))
    print(macs)
    print(params)


print(net.parameters())
print(net)
# print(net.state_dict())

count = 0

def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    for index, p in enumerate(net.parameters()):
        print(' p numue is', index, p.shape, p.numel())
        # count += sum(p.numel)
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params

cnn_paras_count(net)


pthfile = '/datasets/cdd_data/resnet50-0676ba61.pth'  # faster_rcnn_ckpt.pth
net = torch.load(pthfile, map_location=torch.device('cpu'))  # 由于模型原本是用GPU保存的，但我这台电脑上没有GPU，需要转化到CPU上

print(type(net))  # 类型是 dict
print(len(net))   # 长度为 3，即存在3个 key-value 键值对


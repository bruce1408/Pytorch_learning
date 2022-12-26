import argparse
from ngraph.parser.onnx_parser import OnnxParser

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm
from nquantizer.converter.ir.graph import IRGraph
from nquantizer.converter.parsers.onnx.onnx_parser import QuantOnnxParser

from nquantizer.quantize import run_numerical_analysis, run_quantizer

def get_imgnet_loader():
    val_dir = "/home/SharedDatasets/imagenet/val/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
                                             batch_size=10,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)
    return val_loader


def get_dataloader():
    return [(torch.randn(input_shape_list[0]), None)]


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self.conv_2 = nn.Conv2d(
            in_channels=32,
            out_channels = 10,
            kernel_size=3, 
            stride=1,
            padding=1
        )
        
    def forward(self, x):
        output = self.conv_1(x)
        output = output.reshape(1, 32, 4, 112, 112)
        output = output.permute(0, 1, 3, 4, 2)
        output = output.reshape(1, 32, 224, 224)
        output = self.conv_2(output)
        return output


def test_acc(model, ir_graph, data_loader, device=0):
    correct_1 = 0
    correct_5 = 0
    f_correct_1 = 0
    total = 0
    for data in tqdm(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        output = ir_graph(images)
        _, predict = output.topk(5, 1, True, True)
        predict = predict.t()
        correct = predict.eq(labels.view(1, -1).expand_as(predict))
        correct_1 += correct[:1].flatten().float().sum(0, keepdim=True)
        correct_5 += correct[:5].flatten().float().sum(0, keepdim=True)
        total += labels.shape[0]
        # if total > 2000:
        #     break

    print(float(correct_1) / float(total), float(correct_5) / float(total))


if __name__ == "__main__":
    device = 0
    torch.cuda.set_device(device)
    val_loader = get_imgnet_loader()
    torch_module = ToyModel()

    # torch_module = models.load_static_dict(static_dict)
    

    import io
    buffer = io.BytesIO()

    loader = iter(val_loader)
    input_var = next(loader)[0]
    input_var = input_var.to(torch.float32)[:1]
    torch.onnx.export(torch_module.cpu(), input_var,
                      buffer, opset_version=11, training=True)

    import onnx
    # import onnx.shape_inference
    onnx_model = onnx.load_from_string(buffer.getvalue())
    buffer.close()

    cuda_available = True

    input_var, _ = next(iter(val_loader))
    input_var = input_var[0:1]

    parser = QuantOnnxParser([], device=device)
    IRGraph.node_builder = parser.onnx_pytorch_node(device)

    from nquantizer.converter.optimizers import GraphTransform, GraphFuser
    graph =  OnnxParser(onnx_model,
                            graph_type=IRGraph,
                            perform_simplify=True,
                            dummpy_node=True).n_graph.to(device).eval()

    gt = [t() for t in GraphTransform.all_classes_().values()]
    gf = [f() for f in GraphFuser.all_classes_().values()]
    for opt in gt + gf:
        graph = opt(graph)
    quant_graph = run_quantizer(onnx_model,
                                val_loader,
                                num_batches=200 if cuda_available else 2,
                                input_vars=[input_var],
                                output_dir="./quantizer_output",
                                observer_cfg={"type": "min_max"},
                                device=device)
    

    # test_acc(torch_module, quant_graph, val_loader, device=device)

    #run_compiler(input_dir="/tmp/quantizer_output", output_dir="/tmp/compiler_output", debug=True)

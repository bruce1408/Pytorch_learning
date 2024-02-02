import torch.onnx
import torch
import onnx
import onnxsim
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.norm  = nn.LayerNorm(3)
        self.act   = nn.ReLU()

    def forward(self, x):
        _, _, H, W = x.shape
        L = H * W
        x = self.conv1(x)
        x = x.view(x.shape[0], x.shape[1], L).permute(0, 2, 1)
        x = self.norm(x)
        x = self.act(x)
        return x

def export_onnx_graph():
    input  = torch.Tensor(1, 3, 5, 5).uniform_(-1, 1)
    model  = Model()
    model.eval()

    file   = "./sample-ln-before.onnx"
    torch.onnx.export(
            model         = model,
            args          = (input,),
            f             = file,
            input_names   = ["input0"],
            output_names  = ["output0"],
            opset_version = 12)

    print("\nFinished export {}".format(file))

    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)

    print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, file)


export_onnx_graph()
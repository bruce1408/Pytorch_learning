import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from torch.onnx import register_custom_op_symbolic


def register_custom_op():
    def my_group_norm(g, input, num_groups, scale, bias, eps):
        return g.op("mydomain::testgroupnorm", input, num_groups, scale, bias, epsilon_f=0.)

    # Optional: register custom domain version. If not registered, default version is 1
    # set_custom_domain_version("mydomain", 2)

    register_custom_op_symbolic("mynamespace::custom_group_norm", my_group_norm, 9)


def export_custom_op():
    class CustomModel(torch.nn.Module):
        def forward(self, x, num_groups, scale, bias):
            return torch.ops.mynamespace.custom_group_norm(x, num_groups, scale, bias, 0.)

    X = torch.randn(3, 2, 1, 2)
    num_groups = torch.tensor([2.])
    scale = torch.tensor([1., 1.])
    bias = torch.tensor([0., 0.])
    inputs = (X, num_groups, scale, bias)

    f = './model.onnx'
    torch.onnx.export(CustomModel(), inputs, f,
                      opset_version=9,
                    #   example_outputs=None,
                      input_names=["X", "num_groups", "scale", "bias"], output_names=["Y"],
                      custom_opsets={"mydomain": 1})


if __name__ == "__main__":
    torch.ops.load_library(
        "build/lib.linux-x86_64-cpython-39/custom_group_norm.cpython-39-x86_64-linux-gnu.so")

    register_custom_op()
    export_custom_op()
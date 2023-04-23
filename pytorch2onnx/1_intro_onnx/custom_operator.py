import torch.autograd.Function as Function

""" 
    Pytorch 自定义激活函数，都需要继承 torch.autograd.Function 类,
    其内部需要定义两个静态方法【@staticmethod】:forward & backward
"""

class Exp(Function):

    @staticmethod
    def forward(ctx, input):
        result = input.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result


if __name__ == "__main__":
    
    # 通过调用Apply方法来使用它
    # Use it by calling the apply method:
    exp	   = Exp.apply
    output = Exp.apply(input)

    # 通过 torch.autgrad.gradcheck()检测反向传播的结构是否正确，正确返回 True
    torch.autgrad.gradcheck(func, torch.randn(10, requires_grad=True, dtype=torch.double))
        
    
    # forward方法：定义了输入张量（可以是多个），返回对应的输出（也可以是多个张量）；
    # backward方法：定义了输出的剃度张量（可以是多个，必须和输出张量一一对应），
    # 返回输入张量（可以是多个，必须喝输入张量一一对应）
    # 之所以一一对应，是因为计算图中的每个张量在方向传播的时候，输出张量和输入张量对应的梯度绑定，输入张量和输入张良对应的梯度绑定
    


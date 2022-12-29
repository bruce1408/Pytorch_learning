import warnings
import onnxruntime 
import numpy as np
import torch 
import my_lib 
from torch.onnx import register_custom_op_symbolic

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class MyAddFunction(torch.autograd.Function): 
 
    @staticmethod 
    def forward(ctx, a, b): 
        return my_lib.my_add(a, b) 
 
    @staticmethod 
    def symbolic(g, a, b): 
        two = g.op("Constant", value_t=torch.tensor([2])) 
        a = g.op('Mul', a, two) 
        return g.op('Add', a, b) 
 

my_add = MyAddFunction.apply 
 
class MyAdd(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, a, b): 
        return my_add(a, b) 

 
def export_onnx():
    model = MyAdd() 
    torch.onnx.export(model, (input1, input1), 'my_add.onnx',input_names=['a', 'b']) 
    torch_output = model(input1, input1).detach().numpy() 
    return torch_output


def onnx_runtime():
  
    sess = onnxruntime.InferenceSession('my_add.onnx') 
    ort_input = {'a': input1.numpy(), 'b': input1.numpy()}
    ort_output = sess.run(None, ort_input)[0] 
    return ort_output

if __name__=="__main__":
    
    input1 = torch.rand(1, 3, 10, 10) 
    
    output1 = export_onnx()
    
    output2 = onnx_runtime()
    
    assert np.allclose(output1, output2)
    

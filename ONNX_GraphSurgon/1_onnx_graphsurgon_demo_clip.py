import numpy as np
import onnx
import onnx_graphsurgeon as gs

def generate_model():
    # ==========================================================================
    # 构建模型需要的算子
    # Register functions to make graph generation easier
    # ==========================================================================
    @gs.Graph.register()
    def min(self, *args):
        return self.layer(op="Min", inputs=args, outputs=["min_out"])[0]

    @gs.Graph.register()
    def max(self, *args):
        return self.layer(op="Max", inputs=args, outputs=["max_out"])[0]

    @gs.Graph.register()
    def identity(self, inp):
        return self.layer(op="Identity", inputs=[inp], outputs=["identity_out"])[0]


    # Generate the graph
    graph = gs.Graph()

    graph.inputs = [gs.Variable("input", shape=(4, 4), dtype=np.float32)]

    # # Clip values to [0, 6]
    MIN_VAL = np.array(0, np.float32)
    MAX_VAL = np.array(6, np.float32)

    # ==========================================================================
    # Add identity nodes to make the graph structure a bit more interesting
    # ==========================================================================
    inp = graph.identity(graph.inputs[0])
    max_out = graph.max(graph.min(inp, MAX_VAL), MIN_VAL)
    graph.outputs = [graph.identity(max_out), ]

    # Graph outputs must include dtype information
    graph.outputs[0].to_variable(dtype=np.float32, shape=(4, 4))
    # export onnx model
    onnx.save(gs.export_onnx(graph), "./clip_model.onnx")



def modify_onnx():
    # 这里写成函数是为了，万一还需要这样的替换操作就可以重复利用了
    @gs.Graph.register()
    def replace_with_clip(self, inputs, outputs):
        # Disconnect output nodes of all input tensors
        for inp in inputs:
            inp.outputs.clear()

        # Disconnet input nodes of all output tensors
        for out in outputs:
            out.inputs.clear()

        # Insert the new node.
        return self.layer(op="Clip", inputs=inputs, outputs=outputs)

    # Now we'll do the actual replacement
    # 导入onnx模型
    graph = gs.import_onnx(onnx.load("model.onnx"))

    tmap = graph.tensors()
    # You can figure out the input and output tensors using Netron. In our case:
    # Inputs: [inp, MIN_VAL, MAX_VAL]
    # Outputs: [max_out]
    # 子图的需要断开的输入name和子图需要断开的输出name
    inputs = [tmap["identity_out_0"], tmap["onnx_graphsurgeon_constant_5"], tmap["onnx_graphsurgeon_constant_2"]]
    outputs = [tmap["max_out_6"]]

    # 断开并替换成新的名叫Clip的 OP
    graph.replace_with_clip(inputs, outputs)

    # 删除现在游离的子图
    graph.cleanup().toposort()

    # That's it!
    onnx.save(gs.export_onnx(graph), "./replaced.onnx")
    
    
if __name__ == "__main__":
    # ==========================================================================
    # generate_model()
    # ==========================================================================
    modify_onnx()
    # ==========================================================================
    print("Done!")
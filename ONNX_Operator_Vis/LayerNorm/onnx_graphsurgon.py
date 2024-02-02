import onnx_graphsurgeon as gs
import numpy as np
import onnx

# Register functions to make graph generation easier
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

# Clip values to [0, 6]
MIN_VAL = np.array(0, np.float32)
MAX_VAL = np.array(6, np.float32)

# Add identity nodes to make the graph structure a bit more interesting
inp = graph.identity(graph.inputs[0])
max_out = graph.max(graph.min(inp, MAX_VAL), MIN_VAL)
graph.outputs = [graph.identity(max_out), ]

# Graph outputs must include dtype information
graph.outputs[0].to_variable(dtype=np.float32, shape=(4, 4))

onnx.save(gs.export_onnx(graph), "model.onnx")
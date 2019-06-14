from .optimizers import *
from .custom_operators.base import custom_op, desc_from_tensor
from .augmentation import *
try:
    from .distributed_optimizers import *
except ImportError:
    pass # Internal modules log import warnings as necessary

import deep500 as d5

def from_model(model: d5.ops.OnnxModel, device: d5.DeviceType = d5.GPUDevice()) -> d5.GraphExecutor:
    from .reference_graph_executor import ReferenceGraphExecutor
    return ReferenceGraphExecutor(model, device)

def from_onnx(onnx_file: str, device: d5.DeviceType = d5.GPUDevice()) -> d5.GraphExecutor:
    model = d5.parser.load_and_parse_model(onnx_file)
    return from_model(model, device)
